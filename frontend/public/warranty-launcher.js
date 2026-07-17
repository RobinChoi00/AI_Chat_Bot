/**
 * Osaki / Titan — Setup · Warranty · Delivery launcher (Shopify-safe)
 *
 * Install in Shopify theme (before </body>):
 *
 *   <script
 *     src="https://YOUR-CHAT-HOST/warranty-launcher.js"
 *     data-base-url="https://YOUR-CHAT-HOST"
 *     defer
 *   ></script>
 *
 * Place on the bottom-right, stacked above Tidio (sales chat).
 * data-base-url should point at the Next.js frontend that serves /warranty/embed.
 */
(function () {
  "use strict";

  if (window.__osakiWarrantyLauncherLoaded) return;
  window.__osakiWarrantyLauncherLoaded = true;

  // defer scripts run after parse — document.currentScript is often null
  var script =
    document.currentScript ||
    document.querySelector('script[src*="warranty-launcher"]');
  var baseUrl = (script && script.getAttribute("data-base-url")) || "";
  if (!baseUrl && script && script.src) {
    try {
      baseUrl = new URL(script.src).origin;
    } catch (_e) {
      baseUrl = "";
    }
  }
  baseUrl = (baseUrl || window.location.origin).replace(/\/$/, "");
  if (script && script.src && baseUrl.indexOf("myshopify.com") !== -1) {
    try {
      baseUrl = new URL(script.src).origin.replace(/\/$/, "");
    } catch (_e2) {
      /* keep fallback */
    }
  }

  var TEASER_KEY = "osaki_warranty_launcher_teaser_v2";
  var TEASER_AUTO_HIDE_MS = 4500;
  var Z = 2147483000;

  var styles = document.createElement("style");
  styles.textContent =
    "#osaki-warranty-root{font-family:Inter,system-ui,-apple-system,sans-serif;" +
    "line-height:1.35;-webkit-font-smoothing:antialiased;" +
    "--ow-gold:#c9a962;--ow-gold-light:#e8d5a3;--ow-dark:#0f1419;--ow-dark-mid:#1a2332}" +
    "#osaki-warranty-teaser{position:fixed;right:max(16px,env(safe-area-inset-right));" +
    "bottom:calc(188px + env(safe-area-inset-bottom));max-width:min(200px,calc(100vw - 32px));" +
    "background:#fff;border:1px solid #e5e7eb;border-right:3px solid var(--ow-gold);" +
    "border-radius:10px;padding:8px 12px;box-shadow:0 6px 18px rgba(0,0,0,.12);z-index:" +
    Z +
    ";animation:osakiWarrantyFadeIn .3s ease;pointer-events:none}" +
    "#osaki-warranty-teaser.hiding{animation:osakiWarrantyFadeOut .35s ease forwards}" +
    "#osaki-warranty-teaser p{margin:0;font-size:13px;color:#111827;font-weight:600;" +
    "letter-spacing:.01em;white-space:nowrap}" +
    "#osaki-warranty-btn{position:fixed;right:max(16px,env(safe-area-inset-right));" +
    "bottom:calc(116px + env(safe-area-inset-bottom));display:flex;align-items:center;gap:8px;" +
    "flex-direction:row-reverse;" +
    "border:1.5px solid var(--ow-gold);border-radius:999px;padding:6px 12px 6px 6px;cursor:pointer;" +
    "background:linear-gradient(145deg,var(--ow-dark) 0%,var(--ow-dark-mid) 55%,#243044 100%);" +
    "color:#fff;box-shadow:0 6px 20px rgba(0,0,0,.3),0 0 0 0 rgba(201,169,98,.45);" +
    "z-index:" +
    (Z + 1) +
    ";transition:transform .2s ease,box-shadow .2s ease,border-color .2s ease;" +
    "animation:osakiWarrantyPulse 2.8s ease-in-out infinite}" +
    "#osaki-warranty-btn:hover{transform:translateY(-2px);border-color:var(--ow-gold-light);" +
    "box-shadow:0 10px 28px rgba(0,0,0,.35),0 0 20px rgba(201,169,98,.3);animation:none}" +
    "#osaki-warranty-btn:active{transform:scale(.97)}" +
    "#osaki-warranty-btn .icon-wrap{position:relative;width:40px;height:40px;flex-shrink:0}" +
    "#osaki-warranty-btn .icon-ring{position:absolute;inset:0;border-radius:50%;" +
    "background:linear-gradient(145deg,rgba(201,169,98,.35),rgba(201,169,98,.08));" +
    "border:1px solid rgba(201,169,98,.55);box-shadow:inset 0 1px 0 rgba(255,255,255,.15)}" +
    "#osaki-warranty-btn .icon-main{position:absolute;inset:0;display:flex;align-items:center;" +
    "justify-content:center;font-size:18px;line-height:1}" +
    "#osaki-warranty-btn .icon-badge{position:absolute;right:-1px;bottom:-1px;width:16px;height:16px;" +
    "border-radius:50%;background:linear-gradient(135deg,var(--ow-gold),#a8863a);" +
    "border:1.5px solid var(--ow-dark);display:flex;align-items:center;justify-content:center;" +
    "font-size:9px;line-height:1;box-shadow:0 2px 4px rgba(0,0,0,.2)}" +
    "#osaki-warranty-btn .label{text-align:right;padding-left:2px}" +
    "#osaki-warranty-btn .label strong{display:block;font-size:12px;font-weight:700;" +
    "letter-spacing:.02em;color:#fff}" +
    "#osaki-warranty-btn .label em{display:block;font-style:normal;font-size:10px;" +
    "color:var(--ow-gold-light);margin-top:1px;font-weight:500}" +
    "@media(max-width:639px){#osaki-warranty-btn{padding:5px;border-radius:50%;" +
    "width:50px;height:50px;justify-content:center;gap:0;flex-direction:row;" +
    "bottom:calc(132px + env(safe-area-inset-bottom))}" +
    "#osaki-warranty-teaser{bottom:calc(196px + env(safe-area-inset-bottom));" +
    "padding:6px 10px;border-radius:8px;box-shadow:0 4px 14px rgba(0,0,0,.14)}" +
    "#osaki-warranty-teaser p{font-size:12px}" +
    "#osaki-warranty-btn .label{display:none}" +
    "#osaki-warranty-btn .icon-wrap{width:38px;height:38px}" +
    "#osaki-warranty-btn .icon-main{font-size:20px}}" +
    /* Hide our launcher while Tidio (sales) chat is open — avoids overlap */
    "body.osaki-tidio-chat-open #osaki-warranty-btn," +
    "body.osaki-tidio-chat-open #osaki-warranty-teaser{" +
    "display:none!important;visibility:hidden!important;pointer-events:none!important;" +
    "opacity:0!important}" +
    "#osaki-warranty-panel{position:fixed;inset:0;background:rgba(15,20,25,.55);" +
    "backdrop-filter:blur(2px);z-index:" +
    (Z + 2) +
    ";display:none;align-items:flex-end;justify-content:flex-end;padding:0}" +
    "#osaki-warranty-panel.open{display:flex}" +
    "#osaki-warranty-sheet{width:100%;max-width:430px;height:min(92dvh,720px);" +
    "background:#f9fafb;border-radius:16px 16px 0 0;overflow:hidden;display:flex;" +
    "flex-direction:column;box-shadow:0 -8px 40px rgba(0,0,0,.25);animation:osakiWarrantySlideUp .28s ease;" +
    "position:relative;z-index:2}" +
    /* Mobile: full-screen sheet so Google merchant cards can't sit over the input */
    "@media(max-width:639px){#osaki-warranty-panel{align-items:stretch;justify-content:stretch;padding:0}" +
    "#osaki-warranty-sheet{max-width:none;height:100dvh;max-height:100dvh;border-radius:0;" +
    "box-shadow:none}}" +
    "@media(min-width:640px){#osaki-warranty-panel{align-items:flex-end;justify-content:flex-end;padding:16px;" +
    "padding-right:max(16px,env(safe-area-inset-right));padding-bottom:max(16px,env(safe-area-inset-bottom))}" +
    "#osaki-warranty-sheet{border-radius:16px;height:min(85dvh,680px)}}" +
    "#osaki-warranty-sheet header{display:flex;align-items:center;justify-content:space-between;" +
    "padding:12px 14px;background:linear-gradient(180deg,#fff,#fafafa);" +
    "border-bottom:1px solid #e5e7eb;flex-shrink:0}" +
    "#osaki-warranty-sheet header div strong{display:block;font-size:14px;color:#111827}" +
    "#osaki-warranty-sheet header div span{font-size:11px;color:#6b7280}" +
    "#osaki-warranty-close{border:0;background:#f3f4f6;width:36px;height:36px;border-radius:50%;" +
    "cursor:pointer;font-size:18px;line-height:1;color:#374151}" +
    "#osaki-warranty-frame{flex:1;border:0;width:100%;background:#f9fafb}" +
    "@keyframes osakiWarrantyFadeIn{from{opacity:0;transform:translateY(8px)}" +
    "to{opacity:1;transform:translateY(0)}}" +
    "@keyframes osakiWarrantyFadeOut{from{opacity:1;transform:translateY(0)}" +
    "to{opacity:0;transform:translateY(6px)}}" +
    "@keyframes osakiWarrantySlideUp{from{opacity:0;transform:translateY(24px)}" +
    "to{opacity:1;transform:translateY(0)}}" +
    "@keyframes osakiWarrantyPulse{0%,100%{box-shadow:0 8px 28px rgba(0,0,0,.35),0 0 0 0 rgba(201,169,98,.4)}" +
    "50%{box-shadow:0 8px 28px rgba(0,0,0,.35),0 0 0 10px rgba(201,169,98,0)}}" +
    "body.osaki-warranty-panel-open shopify-shop-app," +
    "body.osaki-warranty-panel-open #shopify-shop-app," +
    "body.osaki-warranty-panel-open [id^='shopify-block-shopify-shop']," +
    "body.osaki-warranty-panel-open iframe[src*='shop.app']," +
    /* Google Shopping / “Top Quality Store” merchant overlays */
    "body.osaki-warranty-panel-open iframe[src*='google.com/shopping']," +
    "body.osaki-warranty-panel-open iframe[src*='shopping.google']," +
    "body.osaki-warranty-panel-open [aria-label*='Top Quality Store' i]," +
    "body.osaki-warranty-panel-open [aria-label*='Top quality store' i]," +
    "body.osaki-warranty-panel-open [data-osaki-hidden-overlay='1']{" +
    "display:none!important;visibility:hidden!important;pointer-events:none!important;" +
    "opacity:0!important;max-height:0!important;overflow:hidden!important}";
  document.head.appendChild(styles);

  if (!document.body) {
    document.addEventListener("DOMContentLoaded", initLauncher);
    return;
  }
  initLauncher();

  function initLauncher() {
  if (document.getElementById("osaki-warranty-root")) return;

  var root = document.createElement("div");
  root.id = "osaki-warranty-root";
  document.body.appendChild(root);

  var teaser = null;

  function hideTeaser() {
    if (!teaser || !teaser.parentNode) return;
    sessionStorage.setItem(TEASER_KEY, "1");
    teaser.classList.add("hiding");
    setTimeout(function () {
      if (teaser && teaser.parentNode) teaser.remove();
      teaser = null;
    }, 350);
  }

  if (!sessionStorage.getItem(TEASER_KEY)) {
    teaser = document.createElement("div");
    teaser.id = "osaki-warranty-teaser";
    teaser.setAttribute("role", "status");
    teaser.setAttribute("aria-live", "polite");
    teaser.innerHTML = "<p>Need Help</p>";
    root.appendChild(teaser);
    setTimeout(hideTeaser, TEASER_AUTO_HIDE_MS);
  }

  var btn = document.createElement("button");
  btn.id = "osaki-warranty-btn";
  btn.type = "button";
  btn.setAttribute("aria-label", "Open setup, warranty and delivery help");
  btn.innerHTML =
    '<span class="icon-wrap" aria-hidden="true">' +
    '<span class="icon-ring"></span>' +
    '<span class="icon-main">🛡️</span>' +
    '<span class="icon-badge">✓</span>' +
    "</span>" +
    '<span class="label"><strong>Setup · Warranty · Delivery</strong>' +
    "<em>Guided help for your chair</em></span>";
  root.appendChild(btn);

  var panel = document.createElement("div");
  panel.id = "osaki-warranty-panel";
  panel.setAttribute("role", "dialog");
  panel.setAttribute("aria-modal", "true");
  panel.setAttribute("aria-label", "Setup, warranty and delivery support");
  panel.innerHTML =
    '<div id="osaki-warranty-sheet">' +
    "<header>" +
    "<div><strong>Setup · Warranty · Delivery</strong>" +
    "<span>Osaki &amp; Titan massage chairs</span></div>" +
    '<button type="button" id="osaki-warranty-close" aria-label="Close">×</button>' +
    "</header>" +
    '<iframe id="osaki-warranty-frame" title="Warranty support chat" loading="lazy"></iframe>' +
    "</div>";
  root.appendChild(panel);

  var frame = panel.querySelector("#osaki-warranty-frame");
  var closeBtn = panel.querySelector("#osaki-warranty-close");
  var open = false;
  var SHOP_PANEL_CLASS = "osaki-warranty-panel-open";
  var overlayObserver = null;

  function isWarrantyRoot(el) {
    return !!(el && el.closest && el.closest("#osaki-warranty-root"));
  }

  function looksLikeGoogleMerchantOverlay(el) {
    if (!el || isWarrantyRoot(el)) return false;
    var label = (
      (el.getAttribute &&
        (el.getAttribute("aria-label") ||
          el.getAttribute("title") ||
          el.getAttribute("data-title") ||
          "")) ||
      ""
    ).toLowerCase();
    if (label.indexOf("top quality store") !== -1) return true;
    if (el.tagName === "IFRAME") {
      var src = (el.getAttribute("src") || "").toLowerCase();
      if (
        src.indexOf("google.com/shopping") !== -1 ||
        src.indexOf("shopping.google") !== -1
      ) {
        return true;
      }
    }
    // Prefer card/dialog roots — avoid hiding the whole page.
    var text = ((el.innerText || el.textContent || "") + "")
      .replace(/\s+/g, " ")
      .trim();
    if (text.length < 40 || text.length > 1800) return false;
    if (text.toLowerCase().indexOf("top quality store on google") === -1) {
      return false;
    }
    var role = ((el.getAttribute && el.getAttribute("role")) || "").toLowerCase();
    var style = window.getComputedStyle ? window.getComputedStyle(el) : null;
    var pos = style ? style.position : "";
    if (
      pos === "fixed" ||
      pos === "absolute" ||
      pos === "sticky" ||
      role === "dialog" ||
      role === "complementary" ||
      role === "alertdialog"
    ) {
      return true;
    }
    // Fallback: title-bearing leaf cards often lack fixed positioning on some themes.
    return text.toLowerCase().indexOf("learn more about top quality store") !== -1;
  }

  function hideBlockingOverlays() {
    var nodes = document.body.querySelectorAll(
      "div, section, aside, article, iframe, [role='dialog'], [role='complementary']"
    );
    for (var i = 0; i < nodes.length; i++) {
      var el = nodes[i];
      if (isWarrantyRoot(el)) continue;
      if (!looksLikeGoogleMerchantOverlay(el)) continue;
      el.setAttribute("data-osaki-hidden-overlay", "1");
      el.style.setProperty("display", "none", "important");
      el.style.setProperty("visibility", "hidden", "important");
      el.style.setProperty("pointer-events", "none", "important");
    }
  }

  function restoreBlockingOverlays() {
    var hidden = document.querySelectorAll("[data-osaki-hidden-overlay='1']");
    for (var i = 0; i < hidden.length; i++) {
      var el = hidden[i];
      el.removeAttribute("data-osaki-hidden-overlay");
      el.style.removeProperty("display");
      el.style.removeProperty("visibility");
      el.style.removeProperty("pointer-events");
    }
  }

  function startOverlayWatch() {
    stopOverlayWatch();
    hideBlockingOverlays();
    if (!window.MutationObserver) return;
    overlayObserver = new MutationObserver(function () {
      if (open) hideBlockingOverlays();
    });
    overlayObserver.observe(document.body, {
      childList: true,
      subtree: true,
    });
  }

  function stopOverlayWatch() {
    if (overlayObserver) {
      overlayObserver.disconnect();
      overlayObserver = null;
    }
  }

  function hideShopWidgets() {
    document.body.classList.add(SHOP_PANEL_CLASS);
    startOverlayWatch();
  }

  function showShopWidgets() {
    stopOverlayWatch();
    restoreBlockingOverlays();
    document.body.classList.remove(SHOP_PANEL_CLASS);
  }

  function openPanel() {
    if (open) return;
    open = true;
    closeTidioChat();
    hideShopWidgets();
    if (teaser && teaser.parentNode) hideTeaser();
    frame.src =
      baseUrl +
      "/warranty/embed?store=" +
      encodeURIComponent(window.location.hostname || "");
    panel.classList.add("open");
    document.body.style.overflow = "hidden";
    closeBtn.focus();
  }

  function closePanel() {
    if (!open) return;
    open = false;
    panel.classList.remove("open");
    document.body.style.overflow = "";
    showShopWidgets();
    btn.focus();
  }

  var TIDIO_OPEN_CLASS = "osaki-tidio-chat-open";

  // Authoritative state from Tidio open/close events. Stays null until the
  // first event arrives; after that the DOM-heuristic poller must not
  // override it (the heuristic can misread Tidio's iframe and would flip
  // the button back on ~1-2s after Tidio opens).
  var tidioOpenByEvent = null;

  function setTidioOpen(isOpen) {
    if (isOpen) document.body.classList.add(TIDIO_OPEN_CLASS);
    else document.body.classList.remove(TIDIO_OPEN_CLASS);
  }

  function isTidioChatOpen() {
    // Tidio marks the iframe/body when the messenger is expanded.
    var iframe =
      document.getElementById("tidio-chat-iframe") ||
      document.querySelector('iframe[id*="tidio"], iframe[src*="tidio"]');
    if (iframe) {
      var style = window.getComputedStyle ? window.getComputedStyle(iframe) : null;
      if (style && style.display !== "none" && style.visibility !== "hidden") {
        var rect = iframe.getBoundingClientRect();
        // Bubble-only is small; open messenger is a large panel.
        if (rect.width > 280 && rect.height > 360) return true;
      }
    }
    var chatRoot = document.getElementById("tidio-chat");
    if (chatRoot && chatRoot.className && /open|opened|expanded/i.test(String(chatRoot.className))) {
      return true;
    }
    return false;
  }

  function syncTidioVisibility() {
    // Once events have told us the real state, trust them over the heuristic.
    if (tidioOpenByEvent !== null) {
      setTidioOpen(tidioOpenByEvent);
      return;
    }
    setTidioOpen(isTidioChatOpen());
  }

  function closeTidioChat() {
    try {
      if (window.tidioChatApi && typeof window.tidioChatApi.close === "function") {
        window.tidioChatApi.close();
      }
    } catch (_e) {
      /* ignore */
    }
    tidioOpenByEvent = false;
    setTidioOpen(false);
  }

  function bindTidioEvents() {
    function onOpen() {
      tidioOpenByEvent = true;
      setTidioOpen(true);
    }
    function onClose() {
      tidioOpenByEvent = false;
      setTidioOpen(false);
    }
    document.addEventListener("tidioChat-open", onOpen);
    document.addEventListener("tidioChat-close", onClose);
    try {
      if (window.tidioChatApi && typeof window.tidioChatApi.on === "function") {
        window.tidioChatApi.on("open", onOpen);
        window.tidioChatApi.on("close", onClose);
      }
    } catch (_e2) {
      /* ignore */
    }
  }

  if (window.tidioChatApi) {
    bindTidioEvents();
  } else {
    document.addEventListener("tidioChat-ready", bindTidioEvents, { once: true });
    // Fallback if Tidio loads after us without firing ready.
    var tidioTries = 0;
    var tidioTimer = setInterval(function () {
      tidioTries += 1;
      if (window.tidioChatApi || tidioTries > 40) {
        clearInterval(tidioTimer);
        bindTidioEvents();
        syncTidioVisibility();
      }
    }, 500);
  }

  syncTidioVisibility();
  setInterval(syncTidioVisibility, 1200);

  var chatOrigin = "";
  try {
    chatOrigin = baseUrl ? new URL(baseUrl).origin : "";
  } catch (_originErr) {
    chatOrigin = "";
  }

  window.addEventListener("message", function (e) {
    if (!e.data || e.data.type !== "osaki-warranty-open-link") return;
    if (chatOrigin && e.origin !== chatOrigin) return;
    var url = String(e.data.url || "").trim();
    if (!url || !/^https:\/\//i.test(url)) return;
    window.open(url, "_blank", "noopener,noreferrer");
  });

  btn.addEventListener("click", openPanel);
  closeBtn.addEventListener("click", closePanel);
  panel.addEventListener("click", function (e) {
    if (e.target === panel) closePanel();
  });
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && open) closePanel();
  });
  }
})();
