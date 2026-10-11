(function () {
  "use strict";
  document.documentElement.classList.add("js");
  var items = document.querySelectorAll(".reveal");
  var show = function (el) { el.classList.add("is-visible"); };
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches || !("IntersectionObserver" in window)) {
    items.forEach(show);
    return;
  }
  var observer = new IntersectionObserver(function (entries) {
    entries.forEach(function (entry) {
      if (!entry.isIntersecting) return;
      var el = entry.target;
      window.setTimeout(function () { show(el); }, Number(el.dataset.revealDelay || 0));
      observer.unobserve(el);
    });
  }, { threshold: 0.12, rootMargin: "0px 0px -6% 0px" });
  var groups = new Map();
  items.forEach(function (el) {
    var count = groups.get(el.parentElement) || 0;
    el.dataset.revealDelay = Math.min(count * 70, 420);
    groups.set(el.parentElement, count + 1);
    observer.observe(el);
  });
})();
