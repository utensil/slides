// function toArray( o ) {
//   return Array.prototype.slice.call( o );
// }

// function isExternalLink(url) {
//   return /:\/\//.test(url);
// }

// function openExternalLinksInNewTab(selector) {
//   var anchors = toArray(document.querySelectorAll(selector ? selector : 'a'));

//   anchors.forEach(function(element) {
//     if(isExternalLink(element.getAttribute('href'))) {
//       element.setAttribute('target', '_blank');
//     }
//   });
// }

// document.addEventListener("DOMContentLoaded", function(event) {
//   openExternalLinksInNewTab();
// });

// console.log("Hello");

// var markdownRenderer = new marked.Renderer();

// markdownRenderer.em = function (str) {
//   return '_' + str + '_';
// };

// marked.setOptions({
//   renderer: markdownRenderer
// });

Reveal.addEventListener('ready', function () {

  head.load('./_assets/js/reveal-code-focus-modified.js', function () {
    console.log(window);
    console.log(window.hljs);
    RevealCodeFocus();
  });
});

Reveal.addEventListener('slidechanged', function (event) {
  console.log(event);
  var cur = event.currentSlide;
  var url = cur.getAttribute('data-background-iframe');

  var iframeSource = document.querySelector('.iframe-source');
  if (iframeSource == null) {
    iframeSource = document.createElement('div');
    iframeSource.className = 'iframe-source';
    iframeSource.style.display = "none";
    document.body.appendChild(iframeSource);
  }

  if(/^(https?:)?\/\//.test(url)) {
    iframeSource.innerHTML = '<div class="iframe-source">Source: <a target="_blank" href="' + url + '">' 
                              + url + '</a></div>';
    iframeSource.style.display = "block";
  } else {
    iframeSource.innerHTML = "";
    iframeSource.style.display = "none";
  }
})

/*
Reveal.configure({
  showNotes: true,
  previewLinks: true,
});
*/
