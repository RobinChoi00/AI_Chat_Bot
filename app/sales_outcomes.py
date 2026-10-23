"""Purpose-specific observations; silence does not establish abandonment."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import timedelta
import json

IDLE_AFTER_HOURS = 24
QUESTION_STAGES = (
    ('ask_height', 'Height'), ('ask_goal', 'Primary goal'),
    ('ask_doorway', 'Doorway width'), ('ask_doorway_fit', 'Assembled vs disassembled'),
    ('recommend', 'Recommendation shown'),
)


def _tools(message):
    try:
        value = json.loads(message.tools_used or '[]')
        return {item for item in value if isinstance(item, str)} if isinstance(value, list) else set()
    except (TypeError, ValueError):
        return set()


def _stage(message):
    return next((tool[6:] for tool in _tools(message) if isinstance(tool, str) and tool.startswith('stage:')), '')


def _answered(message):
    tools = _tools(message)
    return message.role == 'assistant' and not message.handoff and not _stage(message).startswith('ask_') and (
        _stage(message) in {'shop', 'recommend'} or
        any(str(tool).startswith('policy.') for tool in tools) or
        'catalog.compare' in tools or
        ('catalog.browse_prices' in tools)
    )


def _rate(part, whole):
    return round(part / whole * 100, 1) if whole else None


def aggregate_outcomes(sessions, messages, leads, feedback, *, end, to_datetime):
    ids = {str(row.session_id) for row in sessions}
    by_session = defaultdict(list)
    for message in messages:
        if str(message.session_id) in ids:
            by_session[str(message.session_id)].append(message)
    # Missing timestamps stay unknown for inactivity. IDs preserve ordering
    # when legacy messages share the same timestamp.
    for rows in by_session.values():
        rows.sort(key=lambda row: (to_datetime(row.created_at) or end, row.id or 0))
    requested = set()
    recommended = set()
    nofit = set()
    handoffs = set()
    explicit_human = set()
    repeated = set()
    answers = set()
    reply_sessions = set()
    state_counts = Counter()
    rated_rows = [row for row in feedback if str(row.session_id) in ids and row.rating in {'helpful', 'not_helpful'}]
    rated_sessions = {str(row.session_id) for row in rated_rows}
    ratings = Counter(row.rating for row in rated_rows)
    assistant_count = 0
    unclear_count = 0
    idle_before = end - timedelta(hours=IDLE_AFTER_HOURS)

    def waiting_state(message):
        at = to_datetime(message.created_at)
        if at is None:
            return 'unknown'
        return 'idle_missing_followup' if at <= idle_before else 'active_waiting'

    for sid in ids:
        rows = by_session.get(sid, [])
        unclear = 0
        for message in rows:
            tools = _tools(message)
            stage = _stage(message)
            if message.intent == 'recommend' or stage in {name for name, _ in QUESTION_STAGES} or tools & {'cases.lookup', 'catalog.recommend', 'cases.nofit'}:
                requested.add(sid)
            if message.role != 'assistant':
                continue
            assistant_count += 1
            reply_sessions.add(sid)
            if tools & {'cases.lookup', 'catalog.recommend'} or stage == 'recommend':
                recommended.add(sid)
            if 'cases.nofit' in tools:
                nofit.add(sid)
            if message.handoff:
                handoffs.add(sid)
            if message.intent == 'human' or 'human.requested' in tools:
                explicit_human.add(sid)
            if message.intent == 'unclear':
                unclear += 1
                unclear_count += 1
            if message.handoff in {'repeated_unresolved', 'answer_unresolved'}:
                repeated.add(sid)
            if _answered(message):
                answers.add(sid)
        if unclear >= 2:
            repeated.add(sid)
        if not rows:
            state_counts['unknown'] += 1
            continue
        last = rows[-1]
        if last.role == 'user':
            state_counts[waiting_state(last)] += 1
        elif last.handoff:
            state_counts['human_requested'] += 1
        elif 'cases.nofit' in _tools(last):
            state_counts['no_verified_match'] += 1
        elif _answered(last):
            state_counts['answer_provided'] += 1
        elif _stage(last).startswith('ask_') or last.intent in {'unclear', 'lead_capture'}:
            state_counts[waiting_state(last)] += 1
        else:
            state_counts['unknown'] += 1

    funnel = []
    for stage, label in QUESTION_STAGES:
        counts = Counter()
        for rows in by_session.values():
            positions = [index for index, message in enumerate(rows) if message.role == 'assistant' and _stage(message) == stage]
            if not positions:
                continue
            counts['reached'] += 1
            position = positions[-1]
            after = rows[position + 1:]
            if stage == 'recommend':
                counts['answer_provided'] += 1
            elif any(message.handoff for message in after if message.role == 'assistant'):
                counts['human_requested'] += 1
            elif any(message.role == 'assistant' and _stage(message) and _stage(message) != stage and (_stage(message).startswith('ask_') or _stage(message) == 'recommend') for message in after):
                counts['continued'] += 1
            elif any(_answered(message) for message in after):
                counts['answer_provided'] += 1
            else:
                counts[waiting_state(rows[-1])] += 1
        funnel.append({
            'stage': stage, 'label': label, 'reached': counts['reached'],
            'advanced': counts['continued'], 'continued': counts['continued'],
            'active_waiting': counts['active_waiting'], 'idle_missing_followup': counts['idle_missing_followup'],
            'answer_provided': counts['answer_provided'], 'human_requested': counts['human_requested'],
            'unknown': counts['unknown'], 'dropped': None, 'drop_off_pct': None,
            'measurement': 'Observed latest occurrence of this stage per conversation; inactivity is not proven abandonment.',
        })

    cohort_leads = [row for row in leads if str(row.session_id) in ids]
    delivery = Counter(str(row.forwarded or 'unknown').lower() for row in cohort_leads)
    lead_sessions = {str(row.session_id) for row in cohort_leads}
    completed = recommended | nofit
    return {
        'recommendation': {
            'requested_sessions': len(requested), 'denominator_sessions': len(requested),
            'recommended_sessions': len(recommended & requested),
            'no_verified_match_sessions': len(nofit & requested),
            'completed_sessions': len(completed & requested),
            'completion_rate_pct': _rate(len(completed & requested), len(requested)),
            'note': 'Recommendation or an honest no-verified-match result among conversations with a stored recommendation intent/stage; not a purchase or confirmed customer resolution.',
        },
        'human': {
            'requested_sessions': len(handoffs), 'explicit_requested_sessions': len(explicit_human),
            'request_rate_pct': _rate(len(handoffs), len(ids)),
            'staff_confirmed_sessions': None, 'staff_confirmation_status': 'unmeasured',
            'note': 'A request/redirect is not proof that staff joined or replied.',
        },
        'follow_up': {
            'lead_sessions': len(lead_sessions), 'lead_records': len(cohort_leads),
            'pending': delivery['pending'], 'sent': delivery['sent'], 'failed': delivery['failed'],
            'unknown': sum(value for key, value in delivery.items() if key not in {'pending', 'sent', 'failed'}),
            'submission_failure_rate_pct': _rate(delivery['failed'], delivery['sent'] + delivery['failed']),
            'staff_confirmed': None,
            'note': 'Sent means SMTP accepted submission. Inbox delivery, shopper receipt and staff follow-up are unconfirmed.',
        },
        'conversation': {
            **{key + '_sessions': state_counts[key] for key in ('answer_provided', 'active_waiting', 'idle_missing_followup', 'human_requested', 'no_verified_match', 'unknown')},
            'idle_after_hours': IDLE_AFTER_HOURS,
            'note': 'An answer shown may be a normal one-turn exit; it is not verified resolution. Missing follow-up after inactivity is not confirmed abandonment.',
        },
        'quality': {
            'repeated_unclear_sessions': len(repeated), 'repeated_unclear_rate_pct': _rate(len(repeated), len(reply_sessions)),
            'unclear_turns': unclear_count, 'assistant_turns': assistant_count,
            'helpful': ratings['helpful'], 'not_helpful': ratings['not_helpful'], 'rated': len(rated_rows),
            'helpful_rate_pct': _rate(ratings['helpful'], len(rated_rows)),
            'rated_sessions': len(rated_sessions), 'response_sessions': len(reply_sessions),
            'unrated_sessions': len(reply_sessions - rated_sessions),
            'rating_coverage_pct': _rate(len(rated_sessions & reply_sessions), len(reply_sessions)),
        },
    }, funnel
