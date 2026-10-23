"""Read-only unified history of every stored Warranty, Sales and general chat.

Source counts apply the domain filter before the optional source selection.
General legacy chat rows without a usable session are individual conversations.
No trace excerpts, evidence paths, or collected-data bags are returned.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from fastapi import APIRouter, Header, HTTPException, Path, Query
from sqlalchemy import Integer, MetaData, String, Table, case, cast, false, func, inspect, literal, or_, select, true, union_all

from admin_auth import require_admin_key
from warranty_models import warranty_db_session

router = APIRouter(tags=["admin-history"])
Source = Literal["warranty", "sales", "general"]
_SOURCES = ("warranty", "sales", "general")


def _domain(column):
    return func.coalesce(func.nullif(func.trim(column), ""), "unknown")


def _present(column):
    return column.is_not(None) & (func.trim(column) != "")


def _time_order(column):
    """Compare naive Chicago DB dates with aware UTC timeline dates correctly."""
    text = cast(column, String)
    year = func.strftime("%Y", column)
    spring = func.datetime(func.printf("%s-03-01", year), "weekday 0", "+7 days", "+2 hours")
    autumn = func.datetime(func.printf("%s-11-01", year), "weekday 0", "+2 hours")
    local = func.datetime(column)
    dst = (local >= spring) & (local < autumn)
    aware = or_(text.like("%Z"), text.like("%+__:__"), text.like("%-__:__"))
    return case((aware, func.julianday(column)), else_=func.julianday(column) + case((dst, 5.0 / 24), else_=6.0 / 24))


def _tables(db):
    bind = db.get_bind()
    existing = set(inspect(bind).get_table_names())
    metadata = MetaData()
    return {name: Table(name, metadata, autoload_with=bind) for name in
            ("warranty_tickets", "warranty_turns", "sales_sessions", "sales_messages", "chat_logs") if name in existing}


def _message(group_id, message_id, role, content, created_at, tie, part=0):
    return select(
        group_id.label("group_id"), message_id.label("id"), literal(role).label("role"),
        cast(content, String).label("content"), cast(created_at, String).label("created_at"),
        _time_order(created_at).label("_time"), tie.label("_tie"), literal(part).label("_part"),
    )


def _warranty_messages(tables):
    tickets = tables.get("warranty_tickets")
    if tickets is None:
        return None
    pieces = []
    turns = tables.get("warranty_turns")
    if turns is not None:
        joined = tickets.join(turns, tickets.c.ticket_id == turns.c.ticket_id)
        for role, content, part in (("assistant", turns.c.node_prompt, 0), ("user", turns.c.customer_answer, 1)):
            pieces.append(_message(tickets.c.id, literal("turn:") + cast(turns.c.id, String) + literal(f":{role}"),
                                   role, content, turns.c.created_at, turns.c.id, part).select_from(joined).where(_present(content)))
    if "collected_data" in tickets.c:
        valid = case((func.json_valid(tickets.c.collected_data), tickets.c.collected_data), else_=literal("{}"))
        timeline = case((func.json_type(valid, "$.chat_timeline") == "array", func.json_extract(valid, "$.chat_timeline")), else_=literal("[]"))
        events = func.json_each(timeline).table_valued("key", "value").alias("timeline_event")
        event = case((func.json_valid(events.c.value), events.c.value), else_=literal("{}"))
        content = func.json_extract(event, "$.text")
        timestamp = func.coalesce(func.json_extract(event, "$.created_at"), tickets.c.updated_at, tickets.c.created_at)
        base = _message(tickets.c.id, literal("timeline:") + cast(tickets.c.id, String) + literal(":") + cast(events.c.key, String),
                        "assistant", content, timestamp, cast(events.c.key, Integer), 2)
        # Role comes from a validated, stored timeline record, never a caller.
        base = base.with_only_columns(
            tickets.c.id.label("group_id"),
            (literal("timeline:") + cast(tickets.c.id, String) + literal(":") + cast(events.c.key, String)).label("id"),
            func.json_extract(event, "$.role").label("role"), cast(content, String).label("content"),
            cast(timestamp, String).label("created_at"), _time_order(timestamp).label("_time"),
            cast(events.c.key, Integer).label("_tie"), literal(2).label("_part"),
        ).select_from(tickets.join(events, true())).where(
            func.json_extract(event, "$.role").in_(("user", "assistant")),
            func.json_type(event, "$.text") == "text", _present(content),
        )
        pieces.append(base)
    return union_all(*pieces).subquery("warranty_history_messages") if pieces else None


def _sales_messages(tables):
    sessions, messages = tables.get("sales_sessions"), tables.get("sales_messages")
    if sessions is None or messages is None:
        return None
    return select(
        sessions.c.id.label("group_id"), cast(messages.c.id, String).label("id"), messages.c.role,
        cast(messages.c.content, String).label("content"), cast(messages.c.created_at, String).label("created_at"),
        _time_order(messages.c.created_at).label("_time"), messages.c.id.label("_tie"), literal(0).label("_part"),
    ).select_from(sessions.join(messages, sessions.c.session_id == messages.c.session_id)).where(
        messages.c.role.in_(("user", "assistant")), _present(messages.c.content),
    ).subquery("sales_history_messages")


def _general_records(logs):
    key = case((_present(logs.c.session_id), literal("session:") + logs.c.session_id),
               else_=literal("row:") + cast(logs.c.id, String))
    domain = _domain(logs.c.domain)
    return select(logs, func.min(logs.c.id).over(partition_by=(key, domain)).label("group_id"),
                  domain.label("_domain")).subquery("general_history_records")


def _general_messages(tables):
    logs = tables.get("chat_logs")
    if logs is None:
        return None
    records = _general_records(logs)
    pieces = []
    for role, content, part in (("user", records.c.user_query, 0), ("assistant", records.c.bot_response, 1)):
        pieces.append(_message(records.c.group_id, literal("log:") + cast(records.c.id, String) + literal(f":{role}"),
                               role, content, records.c.created_at, records.c.id, part).where(_present(content)))
    return union_all(*pieces).subquery("general_history_messages")


def _summary(source, table, messages):
    counts = (select(messages.c.group_id, func.count().label("message_count")).group_by(messages.c.group_id).subquery()
              if messages is not None else None)
    statement = select(
        literal(source).label("source"), cast(table.c.id, String).label("record_id"),
        (table.c.ticket_id if source == "warranty" else literal(None, String)).label("ticket_id"),
        table.c.session_id.label("session_id"), _domain(table.c.domain).label("domain"),
        func.coalesce(table.c.status, "unknown").label("status"),
        cast(table.c.created_at, String).label("created_at"), cast(table.c.updated_at, String).label("updated_at"),
        (func.coalesce(counts.c.message_count, 0) if counts is not None else literal(0)).label("message_count"),
        ((literal("Warranty: ") + func.coalesce(func.nullif(table.c.model_name, ""), "unspecified model") + literal(" · ") +
          func.coalesce(func.nullif(table.c.issue_type, ""), "consultation")) if source == "warranty" else
         literal("Sales: ") + func.coalesce(func.nullif(table.c.last_intent, ""), "consultation")).label("title"),
        _time_order(table.c.created_at).label("_time"),
    )
    return statement.select_from(table.outerjoin(counts, table.c.id == counts.c.group_id)) if counts is not None else statement


def _general_summary(logs, messages):
    records = _general_records(logs)
    grouped = select(
        records.c.group_id, func.min(records.c.created_at).label("created_at"), func.max(records.c.created_at).label("updated_at"),
        func.sum(case((_present(records.c.user_query), 1), else_=0) + case((_present(records.c.bot_response), 1), else_=0)).label("message_count"),
    ).group_by(records.c.group_id).subquery("general_history_groups")
    return select(
        literal("general").label("source"), cast(grouped.c.group_id, String).label("record_id"), literal(None, String).label("ticket_id"),
        logs.c.session_id.label("session_id"), _domain(logs.c.domain).label("domain"), literal("recorded").label("status"),
        cast(grouped.c.created_at, String).label("created_at"), cast(grouped.c.updated_at, String).label("updated_at"), grouped.c.message_count,
        func.substr(func.coalesce(func.nullif(logs.c.user_query, ""), "General chat"), 1, 100).label("title"),
        _time_order(grouped.c.created_at).label("_time"),
    ).select_from(grouped.join(logs, grouped.c.group_id == logs.c.id))


def _queries(tables):
    messages = {"warranty": _warranty_messages(tables), "sales": _sales_messages(tables), "general": _general_messages(tables)}
    summaries = []
    for source, name in (("warranty", "warranty_tickets"), ("sales", "sales_sessions")):
        if name in tables:
            summaries.append(_summary(source, tables[name], messages[source]))
    if "chat_logs" in tables:
        summaries.append(_general_summary(tables["chat_logs"], messages["general"]))
    if not summaries:
        empty = select(literal("").label("source"), literal("").label("record_id"), literal(None).label("ticket_id"),
                       literal(None).label("session_id"), literal("unknown").label("domain"), literal("unknown").label("status"),
                       literal(None).label("created_at"), literal(None).label("updated_at"), literal(0).label("message_count"),
                       literal("").label("title"), literal(None).label("_time")).where(false())
        summaries.append(empty)
    return union_all(*summaries).subquery("all_history"), messages


def _iso(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).isoformat()
    except ValueError:
        return None


def _summary_dict(row):
    return {key: (_iso(row[key]) if key in {"created_at", "updated_at"} else row[key]) for key in
            ("source", "record_id", "ticket_id", "session_id", "domain", "status", "created_at", "updated_at", "message_count", "title")}


@router.get("/admin/history")
async def list_admin_history(source: Optional[Source] = None, domain: Optional[str] = Query(None, max_length=253),
                             limit: int = Query(100, ge=1, le=200), offset: int = Query(0, ge=0),
                             x_admin_key: Optional[str] = Header(default=None)):
    require_admin_key(x_admin_key)
    with warranty_db_session() as db:
        history, _ = _queries(_tables(db))
        selected = select(history)
        if domain:
            selected = selected.where(history.c.domain.contains(domain, autoescape=True))
        filtered = selected.subquery("filtered_history")
        counts = {name: 0 for name in _SOURCES}
        for name, count in db.execute(select(filtered.c.source, func.count()).group_by(filtered.c.source)):
            counts[name] = count
        statement = select(filtered)
        if source:
            statement = statement.where(filtered.c.source == source)
        total = counts[source] if source else sum(counts.values())
        rows = db.execute(statement.order_by(filtered.c._time.desc(), filtered.c.source, cast(filtered.c.record_id, Integer).desc())
                          .limit(limit).offset(offset)).mappings().all()
    return {"total": total, "offset": offset, "limit": limit, "counts": counts, "items": [_summary_dict(row) for row in rows]}


@router.get("/admin/history/{source}/{record_id}")
async def get_admin_history(source: Source, record_id: int = Path(..., ge=1),
                            limit: int = Query(100, ge=1, le=200), offset: int = Query(0, ge=0),
                            x_admin_key: Optional[str] = Header(default=None)):
    require_admin_key(x_admin_key)
    with warranty_db_session() as db:
        history, sources = _queries(_tables(db))
        row = db.execute(select(history).where(history.c.source == source, history.c.record_id == str(record_id))).mappings().first()
        if row is None:
            raise HTTPException(status_code=404, detail="Stored conversation not found.")
        messages = sources[source]
        total = int(row["message_count"])
        rows = [] if messages is None else db.execute(select(messages).where(messages.c.group_id == record_id)
            .order_by(messages.c._time, messages.c._tie, messages.c._part, messages.c.id).limit(limit).offset(offset)).mappings().all()
    return {"conversation": _summary_dict(row), "messages": [
        {"id": message["id"], "role": message["role"], "content": message["content"], "created_at": _iso(message["created_at"])}
        for message in rows
    ], "total": total, "offset": offset, "limit": limit}
