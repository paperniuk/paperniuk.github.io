/* Refresh the GitHub star counts shown next to projects. The numbers in the
   HTML are the fallback; if the API is unreachable or rate limited they stay. */
(function () {
  "use strict";
  var TTL_MS = 60 * 60 * 1000;
  var badges = document.querySelectorAll(".stars[data-repo]");
  if (!badges.length || !window.fetch) return;

  function apply(badge, count) {
    var n = badge.querySelector(".stars-n");
    if (typeof count !== "number" || !n) return;
    n.textContent = String(count);
    badge.hidden = count < 1;
  }
  function cached(repo) {
    try {
      var hit = JSON.parse(sessionStorage.getItem("stars:" + repo));
      if (hit && Date.now() - hit.t < TTL_MS) return hit.n;
    } catch (e) { /* storage unavailable */ }
    return null;
  }
  function remember(repo, n) {
    try { sessionStorage.setItem("stars:" + repo, JSON.stringify({ t: Date.now(), n: n })); } catch (e) { /* ignore */ }
  }

  badges.forEach(function (badge) {
    var repo = badge.getAttribute("data-repo");
    var hit = cached(repo);
    if (hit !== null) { apply(badge, hit); return; }
    fetch("https://api.github.com/repos/" + repo, { headers: { Accept: "application/vnd.github+json" } })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (data) {
        if (!data || typeof data.stargazers_count !== "number") return;
        remember(repo, data.stargazers_count);
        apply(badge, data.stargazers_count);
      })
      .catch(function () { /* keep the static number */ });
  });
})();
