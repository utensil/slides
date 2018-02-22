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

head.load('./_assets/css/vendor/mermaid.forest.css');
head.load('./_assets/js/vendor/mermaidAPI.js', function () {
  mermaidAPI.initialize({
    startOnLoad: false,
    cloneCssStyles: false,
    sequenceDiagram: {
      height: 30,
      mirrorActors: false
    }
  });
});

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
});

function renderMermaid(cur) {
  var diagramCodeTag = cur.querySelector('code.lang-mermaid');
  var renderedDiagram = cur.querySelector('.mermaidSvg');

  if(diagramCodeTag != null && mermaidAPI != null) {
    // console.log(diagramCodeTag);
    var diagramSource = diagramCodeTag.textContent;
    // console.log(diagramSource);

    var id = Math.floor(Math.random() * 1000).toString();

    mermaidAPI.render('mermaid-diagram-' + id, diagramSource, function (svgCode, bindFunctions) {
      // console.log(svgCode);
    
      var svgDiv = document.createElement('div');
      svgDiv.className = 'mermaidSvg';
      svgDiv.innerHTML = svgCode;

      cur.insertBefore(svgDiv, diagramCodeTag.parentNode);

      diagramCodeTag.style.display = "none";

      if(renderedDiagram != null) {
        renderedDiagram.remove();
      }
    });
  }
}

Reveal.addEventListener('slidechanged', function (event) {
  var cur = event.currentSlide;
  renderMermaid(cur);
});

if (window.location.search.match( /print-pdf/gi )) {
  Reveal.addEventListener('ready', function () {
    var slides = document.querySelectorAll('.reveal .slides section');
    slides.forEach(function (cur) {
      renderMermaid(cur);

      var codeComments = cur.querySelectorAll('.fragment[data-code-focus]');

      if (codeComments) {
        var codeFocus = cur.querySelectorAll('code.focus');
        codeFocus.forEach(function (c) {
            c.style.zoom = 0.5;
        });
      }
      
      codeComments.forEach(function (codeComment) {
        var codeLineSpec = codeComment.getAttribute('data-code-focus');
        codeComment.classList.remove('fragment');
        codeComment.style.zoom = 1 / (codeComments.length || 2) * 2;
        var codeLineSpecSpan = document.createElement('span');
        codeLineSpecSpan.textContent = 'line ' + codeLineSpec + ': ';
        codeLineSpecSpan.style.cssFloat = 'left';
        codeLineSpecSpan.style.marginLeft = '10%';
        codeComment.insertBefore(codeLineSpecSpan, codeComment.firstChild);
      });

      if (cur.hasAttribute('data-background-iframe')) {
        // console.log(cur);

        var iframeSource = document.createElement('div');
        iframeSource.className = 'iframe-source';

        var url = cur.getAttribute('data-background-iframe');

        var maxLen = 100;

        iframeSource.innerHTML = 'Source: <a target="_blank" href="' + url + '">' 
                              + ( url.length > maxLen ? (url.substr(0, maxLen) + '...') : url) + '</a>';
        iframeSource.style.display = "block";
        cur.appendChild(iframeSource);
      }
      
    });
  });
}
