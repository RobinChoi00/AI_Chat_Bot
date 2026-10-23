/* Explicit storefront action adapter. This does not infer purchases or consent. */
(function () {
  "use strict";
  if (window.SalesAnalytics) return;

  var storageKey = "osaki-sales-analytics-session-v1";
  var context = null;
  var options = { enabled: false, persistSession: false, apiBase: "", domain: "" };
  var pending = Promise.resolve();
  var startedSessions = Object.create(null);

  function sessionIsValid(value) {
    return value && /^[A-Za-z0-9:_.-]{1,200}$/.test(value.sessionId || "") &&
      typeof value.eventToken === "string" && value.eventToken.length > 0 && value.eventToken.length <= 1200;
  }

  function saveContext() {
    if (!options.persistSession) return;
    try {
      if (context) window.sessionStorage.setItem(storageKey, JSON.stringify(context));
      else window.sessionStorage.removeItem(storageKey);
    } catch (_) { /* Analytics must not interrupt a storefront action. */ }
  }

  function setSession(sessionId, eventToken, trackingId) {
    var next = { sessionId: sessionId, eventToken: eventToken, domain: options.domain };
    if (!options.enabled || !sessionIsValid(next)) return false;
    context = next;
    var key = next.domain + "|" + next.sessionId;
    if (window.crypto && window.crypto.randomUUID) {
      context.trackingId = startedSessions[key] || trackingId || window.crypto.randomUUID();
    }
    saveContext();
    if (!startedSessions[key] && context.trackingId) {
      startedSessions[key] = context.trackingId;
      track("tracking_started", null, context.trackingId);
    }
    return true;
  }

  function init(config) {
    config = config || {};
    options.enabled = config.enabled === true;
    options.persistSession = config.persistSession === true;
    options.domain = config.domain || window.location.hostname;
    options.apiBase = (config.apiBase || "").replace(/\/$/, "");
    context = null;
    if (!options.enabled) {
      clear();
      return false;
    }
    if (config.sessionId && config.eventToken) return setSession(config.sessionId, config.eventToken);
    if (options.persistSession) {
      try {
        var prior = JSON.parse(window.sessionStorage.getItem(storageKey) || "null");
        if (prior && prior.domain === options.domain) return setSession(prior.sessionId, prior.eventToken, prior.trackingId);
      } catch (_) { /* Invalid or unavailable tab storage is safely ignored. */ }
    }
    return false;
  }

  function adoptResponse(response) {
    return !!response && setSession(response.session_id, response.analytics_token);
  }

  function clear() {
    context = null;
    try { window.sessionStorage.removeItem(storageKey); } catch (_) { /* No tab storage is also safe. */ }
  }

  function productUrl(raw) {
    if (!raw) return null;
    try {
      var url = new URL(raw, window.location.origin);
      if (url.protocol !== "https:" || !/^\/products\/[a-z0-9][a-z0-9-]{0,199}\/?$/.test(url.pathname)) return null;
      // Campaign strings, customer identifiers, and fragments never enter analytics.
      return url.origin + url.pathname;
    } catch (_) { return null; }
  }

  function track(kind, rawProductUrl, eventId) {
    if (!options.enabled || !context) return Promise.resolve(false);
    if (["tracking_started", "product_click", "add_to_cart", "checkout_started"].indexOf(kind) < 0) return Promise.resolve(false);
    var url = productUrl(rawProductUrl);
    if ((rawProductUrl && !url) || (kind === "product_click" && !url)) return Promise.resolve(false);
    if (!window.crypto || !window.crypto.randomUUID) return Promise.resolve(false);
    var event = {
      event_id: eventId || window.crypto.randomUUID(),
      session_id: context.sessionId,
      event_token: context.eventToken,
      domain: context.domain,
      event_type: kind,
      product_url: url,
    };
    var endpoint = options.apiBase + "/api/v1/sales/events";
    // Serialize observed actions so a later checkout does not overtake the cart.
    pending = pending.catch(function () { return false; }).then(function () {
      return window.fetch(endpoint, {
        method: "POST",
        credentials: "omit",
        keepalive: true,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(event),
      }).then(function (response) { return response.ok; }).catch(function () { return false; });
    });
    return pending;
  }

  document.addEventListener("click", function (event) {
    if (!event.target || !event.target.closest) return;
    var link = event.target.closest("a[data-sales-product]");
    if (link) track("product_click", link.href);
  });

  window.SalesAnalytics = {
    init: init,
    adoptResponse: adoptResponse,
    setSession: setSession,
    trackProductClick: function (url) { return track("product_click", url); },
    recordCart: function (url) { return track("add_to_cart", url); },
    recordCheckout: function () { return track("checkout_started"); },
    clear: clear,
  };
})();
