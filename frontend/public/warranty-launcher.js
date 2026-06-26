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

  var TEASER_KEY = "osaki_warranty_launcher_teaser_v1";
  var Z = 2147483000;

  var styles = document.createElement("style");
  styles.textContent =
    "#osaki-warranty-root{font-family:Inter,system-ui,-apple-system,sans-serif;" +
    "line-height:1.35;-webkit-font-smoothing:antialiased;" +
    "--ow-gold:#c9a962;--ow-gold-light:#e8d5a3;--ow-dark:#0f1419;--ow-dark-mid:#1a2332}" +
    "#osaki-warranty-teaser{position:fixed;right:max(16px,env(safe-area-inset-right));" +
    "bottom:calc(188px + env(safe-area-inset-bottom));max-width:min(280px,calc(100vw - 32px));" +
    "background:#fff;border:1px solid #e5e7eb;border-right:4px solid var(--ow-gold);" +
    "border-radius:14px;padding:12px 14px;box-shadow:0 12px 32px rgba(0,0,0,.14);z-index:" +
    Z +
    ";animation:osakiWarrantyFadeIn .35s ease}" +
    "#osaki-warranty-teaser p{margin:0 0 8px;font-size:13px;color:#111827;font-weight:600}" +
    "#osaki-warranty-teaser span{display:block;font-size:12px;color:#4b5563;font-weight:400}" +
    "#osaki-warranty-teaser button{margin-top:8px;border:0;background:transparent;" +
    "color:#6b7280;font-size:11px;cursor:pointer;padding:0}" +
    "#osaki-warranty-btn{position:fixed;right:max(16px,env(safe-area-inset-right));" +
    "bottom:calc(110px + env(safe-area-inset-bottom));display:flex;align-items:center;gap:10px;" +
    "flex-direction:row-reverse;" +
    "border:2px solid var(--ow-gold);border-radius:999px;padding:8px 16px 8px 8px;cursor:pointer;" +
    "background:linear-gradient(145deg,var(--ow-dark) 0%,var(--ow-dark-mid) 55%,#243044 100%);" +
    "color:#fff;box-shadow:0 8px 28px rgba(0,0,0,.35),0 0 0 0 rgba(201,169,98,.45);" +
    "z-index:" +
    (Z + 1) +
    ";transition:transform .2s ease,box-shadow .2s ease,border-color .2s ease;" +
    "animation:osakiWarrantyPulse 2.8s ease-in-out infinite}" +
    "#osaki-warranty-btn:hover{transform:translateY(-2px);border-color:var(--ow-gold-light);" +
    "box-shadow:0 14px 36px rgba(0,0,0,.4),0 0 24px rgba(201,169,98,.35);animation:none}" +
    "#osaki-warranty-btn:active{transform:scale(.97)}" +
    "#osaki-warranty-btn .icon-wrap{position:relative;width:48px;height:48px;flex-shrink:0}" +
    "#osaki-warranty-btn .icon-ring{position:absolute;inset:0;border-radius:50%;" +
    "background:linear-gradient(145deg,rgba(201,169,98,.35),rgba(201,169,98,.08));" +
    "border:1px solid rgba(201,169,98,.55);box-shadow:inset 0 1px 0 rgba(255,255,255,.15)}" +
    "#osaki-warranty-btn .icon-main{position:absolute;inset:0;display:flex;align-items:center;" +
    "justify-content:center;font-size:22px;line-height:1}" +
    "#osaki-warranty-btn .icon-badge{position:absolute;right:-2px;bottom:-2px;width:20px;height:20px;" +
    "border-radius:50%;background:linear-gradient(135deg,var(--ow-gold),#a8863a);" +
    "border:2px solid var(--ow-dark);display:flex;align-items:center;justify-content:center;" +
    "font-size:11px;line-height:1;box-shadow:0 2px 6px rgba(0,0,0,.25)}" +
    "#osaki-warranty-btn .label{text-align:right;padding-left:2px}" +
    "#osaki-warranty-btn .label strong{display:block;font-size:13px;font-weight:700;" +
    "letter-spacing:.02em;color:#fff}" +
    "#osaki-warranty-btn .label em{display:block;font-style:normal;font-size:11px;" +
    "color:var(--ow-gold-light);margin-top:2px;font-weight:500}" +
    "@media(max-width:639px){#osaki-warranty-btn{padding:6px;border-radius:50%;" +
    "width:58px;height:58px;justify-content:center;gap:0;flex-direction:row;" +
    "bottom:calc(128px + env(safe-area-inset-bottom))}" +
    "#osaki-warranty-teaser{bottom:calc(200px + env(safe-area-inset-bottom))}" +
    "#osaki-warranty-btn .label{display:none}" +
    "#osaki-warranty-btn .icon-wrap{width:44px;height:44px}" +
    "#osaki-warranty-btn .icon-main{font-size:24px}}" +
    "#osaki-warranty-panel{position:fixed;inset:0;background:rgba(15,20,25,.55);" +
    "backdrop-filter:blur(2px);z-index:" +
    (Z + 2) +
    ";display:none;align-items:flex-end;justify-content:flex-end;padding:0}" +
    "#osaki-warranty-panel.open{display:flex}" +
    "#osaki-warranty-sheet{width:100%;max-width:430px;height:min(92dvh,720px);" +
    "background:#f9fafb;border-radius:16px 16px 0 0;overflow:hidden;display:flex;" +
    "flex-direction:column;box-shadow:0 -8px 40px rgba(0,0,0,.25);animation:osakiWarrantySlideUp .28s ease}" +
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
    "@keyframes osakiWarrantySlideUp{from{opacity:0;transform:translateY(24px)}" +
    "to{opacity:1;transform:translateY(0)}}" +
    "@keyframes osakiWarrantyPulse{0%,100%{box-shadow:0 8px 28px rgba(0,0,0,.35),0 0 0 0 rgba(201,169,98,.4)}" +
    "50%{box-shadow:0 8px 28px rgba(0,0,0,.35),0 0 0 10px rgba(201,169,98,0)}}";
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
  if (!sessionStorage.getItem(TEASER_KEY)) {
    teaser = document.createElement("div");
    teaser.id = "osaki-warranty-teaser";
    teaser.innerHTML =
      "<p>🛡️ Setup, warranty &amp; delivery help</p>" +
      "<span>Step-by-step guide for your chair — before you call or email.</span>" +
      '<button type="button" aria-label="Dismiss">Dismiss</button>';
    root.appendChild(teaser);
    teaser.querySelector("button").addEventListener("click", function () {
      sessionStorage.setItem(TEASER_KEY, "1");
      teaser.remove();
      teaser = null;
    });
    setTimeout(function () {
      if (teaser && teaser.parentNode) {
        sessionStorage.setItem(TEASER_KEY, "1");
        teaser.remove();
      }
    }, 12000);
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

  function openPanel() {
    if (open) return;
    open = true;
    if (teaser && teaser.parentNode) {
      sessionStorage.setItem(TEASER_KEY, "1");
      teaser.remove();
      teaser = null;
    }
    frame.src = baseUrl + "/warranty/embed";
    panel.classList.add("open");
    document.body.style.overflow = "hidden";
    closeBtn.focus();
  }

  function closePanel() {
    if (!open) return;
    open = false;
    panel.classList.remove("open");
    document.body.style.overflow = "";
    btn.focus();
  }

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
