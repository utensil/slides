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

// https://stackoverflow.com/a/14570614/200764
var observeDOM = (function(){
  var MutationObserver = window.MutationObserver || window.WebKitMutationObserver,
      eventListenerSupported = window.addEventListener;

  return function(obj, callback){
      if( MutationObserver ){
          // define a new observer
          var obs = new MutationObserver(function(mutations, observer){
              if( mutations[0].addedNodes.length || mutations[0].removedNodes.length )
                  callback();
          });
          // have the observer observe foo for changes in children
          obs.observe( obj, { childList:true, subtree:true });
      }
      else if( eventListenerSupported ){
          obj.addEventListener('DOMNodeInserted', callback, false);
          obj.addEventListener('DOMNodeRemoved', callback, false);
      }
  };
})();

Reveal.addEventListener('ready', function () {

  head.load('./_assets/js/reveal-code-focus-modified.js', function () {
    console.log(window);
    console.log(window.hljs);
    RevealCodeFocus();
  });

  if(window.location.search.match( /print-pdf/gi )) {
    var timeOfLastUpdate = new Date();
    var idleTime = 10000 /* ms */;

    observeDOM(document.querySelector('.reveal'), function () {
      // console.log('observeDOM', Date.now() - timeOfLastUpdate);

      timeOfLastUpdate = Date.now();

      setTimeout(function () {
        if (Date.now() - timeOfLastUpdate > idleTime) {
          var allBgs = document.querySelectorAll('.print-pdf .pdf-page .slide-background');
          
          allBgs.forEach(function (bg) {
            var iframeBg = bg.querySelector('iframe');
            if(iframeBg != null) {
              var url = iframeBg.getAttribute('src');
              iframeSource = document.createElement('div');
              iframeSource.innerHTML = '<div class="iframe-source">Source: <a target="_blank" href="' + url + '">' 
              + url + '</a></div>';
              bg.appendChild(iframeSource);
            }
          });
  
          console.log('iframe source added!');
        }
      }, idleTime);      
    });
  }
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
