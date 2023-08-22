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

// Reveal.initialize({
//   mathjax3: {
//     mathjax: 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js',
//     tex: {
//       inlineMath: [ [ '$', '$' ], [ '\\(', '\\)' ]  ]
//     },
//     options: {
//       skipHtmlTags: [ 'script', 'noscript', 'style', 'textarea', 'pre', 'iframe' ]
//     },
//   },
//   plugins: [ RevealMath.MathJax3 ]
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

// https://stereochro.me/ideas/detecting-broken-images-js
function isImageOk(img) {
  // During the onload event, IE correctly identifies any images that
  // weren't downloaded as not complete. Others should too. Gecko-based
  // browsers act like NS4 in that they report this incorrectly.
  if (!img.complete) {
      return false;
  }

  // However, they do have two very useful properties: naturalWidth and
  // naturalHeight. These give the true size of the image. If it failed
  // to load, either of these should be zero.
  if (typeof img.naturalWidth != "undefined" && img.naturalWidth == 0) {
      return false;
  }

  // No other way of checking: assume it's ok.
  return true;
}

var dependencies = [];

function loadJs(url, callback) {
  const script = document.createElement('script');
  script.src = url;
  script.onload = callback;
  document.head.appendChild(script);
}

function loadCSS(url, callback) {
  var link = document.createElement('link');
  link.rel = 'stylesheet';
  link.type = 'text/css';
  link.href = url;
  link.onload = callback;
  document.getElementsByTagName('head')[0].appendChild(link);
}

loadJs('./_assets/js/vendor/fontawesome-all.min.js');
loadCSS('./_assets/css/vendor/mermaid.forest.css');
loadJs('./_assets/js/vendor/mermaidAPI.js', function () {
  mermaidAPI.initialize({
    startOnLoad: false,
    cloneCssStyles: true,
    sequenceDiagram: {
      height: 30,
      mirrorActors: false
    }
  });
});

var fancyboxOptions = {
  baseClass : 'reveal-fancybox',
  buttons : [
    'slideShow',
    'fullScreen',
    'thumbs',
    'download',
    'zoom',
    'close'
  ],
  smallBtn : false,
  afterShow: function (instance, slide) {
    console.info(instance, slide);
    $(slide.$slide[0]).scrollTop($(slide.$content[0]).offset().top);
  },
  afterClose: function (instance, slide) {
    // console.info( slide.opts.$orig );
    slide.opts.$orig.each(function (i, a) {
      // console.log(a);
      $('svg', a).show();
    });
    Reveal.layout();
  }    
};

function funcyboxifyImages(cur) {
  if (!$) {
    return;
  }

  var imgs = $('img', cur || document);
  imgs.each(function (i, img) {
    if ($(img).parent().is('[data-fancybox]') || $(img).fancybox == null) {
      // console.log('ignored', img);
      return;
    }

    $(img).wrap('<a href="' + $(img).attr('src') + '" data-fancybox="images"></a>');
  });
}

Reveal.on('ready', function (event) {

  loadJs('./_assets/js/reveal-code-focus-modified.js', function () {
    // console.log(window);
    // console.log(window.hljs);
    RevealCodeFocus();
  });

  loadJs('./_assets/js/vendor/jquery-3.3.1.min.js', function() {
    loadCSS('./_assets/css/vendor/jquery.fancybox.min.css', function() {
      loadJs('./_assets/js/vendor/jquery.fancybox.min.js', function () {
          funcyboxifyImages();
          $("[data-fancybox]").fancybox(fancyboxOptions);
      })
    })
  });

  var cur = event.currentSlide;
  decorateSlide(cur, event);
});

Reveal.addEventListener('slidechanged', function (event) {
    var cur = event.currentSlide;    
    decorateSlide(cur, event);
});

function renderIframeSource(cur) {
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
}

function ensureSVGRender() {
  if(window.MathJax) {
    // Get original renderer
    originalRenderer = MathJax.Hub.config.menuSettings.renderer;
    if(originalRenderer !== 'SVG') {
      // MathJax.Hub.Register.StartupHook("End Jax",function () {
      //   var jax = "SVG";
      //   return MathJax.Hub.setRenderer(jax);
      // });
      MathJax.Hub.setRenderer('SVG');
      MathJax.Hub.Queue(
        ["Rerender",MathJax.Hub],
        function() {
          Reveal.layout();
          console.log('Rendering MathJax with SVG...');
        });
    }
  }
}

function decorateSlide(cur, event) {
    // console.log(event);
    console.log(event.indexh, event.indexv);

    ensureSVGRender();

    renderIframeSource(cur);
    renderMermaid(cur);

    // Refresh broken image once
    var curImages = cur.querySelectorAll('img');
    for (var i = 0; i < curImages.length; i++) {
      if (!isImageOk(curImages[i])) {
        curImages[i].src = curImages[i].src + '#' + new Date().getTime();
      }
    }

    // funcyboxifyImages(cur);

    // // Ensure fancybox works
    // if($) {
    //   var fancybox = $("[data-fancybox]", cur);    
    //   if(fancybox.fancybox) {
    //     fancybox.fancybox(fancyboxOptions);
    //   }
    // }
  
    // Ensure the layout is correct
    Reveal.layout();
}

function renderMermaid(cur) {
  var diagramCodeTag = cur.querySelector('code.lang-mermaid');
  var renderedDiagram = cur.querySelector('.mermaidSvg');

  if(diagramCodeTag != null && mermaidAPI != null) {
    // console.log(diagramCodeTag);
    var diagramSource = diagramCodeTag.textContent;
    // console.log(diagramSource);

    var id = Math.floor(Math.random() * 1000).toString();
    var fullId = 'mermaid-diagram-' + id;

    mermaidAPI.render(fullId, diagramSource, function (svgCode, bindFunctions) {
      // console.log(svgCode);
    
      var svgDiv = document.createElement('a');
      svgDiv.className = 'mermaidSvg';
      svgDiv.setAttribute('data-fancybox', 'images');
      svgDiv.setAttribute('data-src', '#' + fullId);

      svgDiv.innerHTML = svgCode;

      cur.insertBefore(svgDiv, diagramCodeTag.parentNode);

      diagramCodeTag.style.display = "none";

      if(renderedDiagram != null) {
        renderedDiagram.remove();
      }

      return;

      var svgElement = cur.querySelector('svg#' + fullId);
      // https://css-tricks.com/using-svg/
      var svgPrefix = 'data:image/svg+xml,';
      svgElement.href = svgPrefix + svgCode.replace(/<br>/g, '<br />');
      // encodeURIComponent(svgCode).replace(/</g, '%3C').replace(/>/g, '%3E')
      // .replace(/#/g, '%23').replace(/"/g, "%22")      
      console.log(svgElement);
      // https://stackoverflow.com/questions/34904306/svg-with-inline-css-to-image-data-uri
      // get svg data
      // var xml = new XMLSerializer().serializeToString(svgElement);
      // // make it base64
      // var svg64 = btoa(xml);
      // var image64 = 'data:image/svg+xml;base64,' + svg64;

      svgDiv = svgElement.parentNode;
      // svgDiv.href = image64;
      var bbox = svgElement.getBBox()
      svgDiv.setAttribute('width', bbox.width);
      svgDiv.setAttribute('height', bbox.height);   
      
      Reveal.layout();
    });
  }
}

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


/* 
var revealConfig = Reveal.getConfig();

revealConfig.dependencies = (revealConfig.dependencies || []).concat(dependencies);

// console.log('Reveal.getConfig():', revealConfig);

revealConfig.markdown = revealConfig.markdown || {};

Object.defineProperty(revealConfig.markdown, "renderer", { get: function () {
    var customRenderer = new marked.Renderer();

    // quick test:
    // customRenderer.heading = function (text, level) {
    //   return `<h${level}>
    //             ${text} ^_^
    //           </h${level}>`;
    // }

    function escape(html, encode) {
      return html
        .replace(!encode ? /&(?!#?\w+;)/g : /&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }

    var origCode = customRenderer.code.bind(customRenderer);
    customRenderer.code = function (code, lang, escaped) {
      if (lang != 'xxx') {
        return origCode(code, lang, escaped);
      } else {
        return '<pre>xxx: TODO</pre>';
      }

      // origCode:
      // if (this.options.highlight) {
      //   var out = this.options.highlight(code, lang);
      //   if (out != null && out !== code) {
      //     escaped = true;
      //     code = out;
      //   }
      // }
    
      // if (!lang) {
      //   return '<pre><code>'
      //     + (escaped ? code : escape(code, true))
      //     + '\n</code></pre>';
      // }
    
      // return '<pre><code class="'
      //   + this.options.langPrefix
      //   + escape(lang, true)
      //   + '">'
      //   + (escaped ? code : escape(code, true))
      //   + '\n</code></pre>\n';
    }

    return customRenderer;
  },
  set: function (v) {
    this._renderer = v;
  },
  enumerable: true
});

*/


