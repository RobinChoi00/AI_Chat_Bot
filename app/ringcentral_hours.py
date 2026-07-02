"""
Warranty phone-line business hours (America/Chicago).

Matches ``WARRANTY_BUSINESS_HOURS`` in config.py:
  Mon–Fri, 10:00 AM – 6:00 PM CST
"""

from __future__ import annotations

from datetime import datetime, time
from typing import Optional

import pytz

_WARRANTY_TZ = pytz.timezone("America/Chicago")
_WARRANTY_OPEN = time(10, 0)
_WARRANTY_CLOSE = time(18, 0)


def is_warranty_business_hours(now: Optional[datetime] = None) -> bool:
    """Return True during Mon–Fri 10:00–17:59:59 America/Chicago."""
    moment = now or datetime.now(_WARRANTY_TZ)
    if moment.tzinfo is None:
        moment = _WARRANTY_TZ.localize(moment)
    else:
        moment = moment.astimezone(_WARRANTY_TZ)

    if moment.weekday() >= 5:
        return False
    current = moment.time()
    return _WARRANTY_OPEN <= current < _WARRANTY_CLOSE
