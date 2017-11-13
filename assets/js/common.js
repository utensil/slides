function toArray( o ) {
  return Array.prototype.slice.call( o );
}

function isExternalLink(url) {
  return /:\/\//.test(url);
}

function openExternalLinksInNewTab(selector) {
  var anchors = toArray(document.querySelectorAll(selector ? selector : 'a'));

  anchors.forEach(function(element) {
    if(isExternalLink(element.getAttribute('href'))) {
      element.setAttribute('target', '_blank');
    }
  });
}

document.addEventListener("DOMContentLoaded", function(event) {
  openExternalLinksInNewTab();
});

/*
Reveal.configure({
  showNotes: true,
  previewLinks: true,
});
*/
