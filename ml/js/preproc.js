var plantuml = require('node-plantuml');
// var Sync = require('sync');
// var deasync = require('deasync');
var fs = require('fs');
var crypto = require('crypto');

// plantuml.useNailgun(); // Activate the usage of Nailgun

function testfindNextCodeInLanguage() {
  var lang = 'lang';
  var md = 'abc\n```lang\nxxxxxx\n```\nasdf\n```lang\nijbieiuh\n```\nsxdede';
  var found = findNextCodeInLanguage(lang, md);
  console.log(md.substring(found.start, found.stop));
  console.log(md.substring(found.outerStart, found.outerStop));

  found = findNextCodeInLanguage(lang, md, found.outerStop);
  console.log(md.substring(found.start, found.stop));
  console.log(md.substring(found.outerStart, found.outerStop));
}

function testReplaceCodeInLanguage() {
  var lang = 'lang';
  var md = 'abc\n```lang\nxxxxxx\n```\nasdf\n```lang\nijbieiuh\n```\nsxdede';

  console.log(replaceCodeInLanguage('lang', md, function (original) {
    return '[[[' + original + ']]]';
  }));

  // markdown = replaceCodeInLanguage('puml', markdown, function (original) {
  //   return plantuml.generate(original, {format: 'svg'});
  // });
}

function findNextCodeInLanguage(lang, str, start) {
  var ret = {};
  start = start || 0;
  var LANG_BEGIN = '```' + lang;
  var LANG_END = '```';

  start = str.indexOf(LANG_BEGIN, start);

  if (start != -1) {
    ret.start = start + LANG_BEGIN.length;
    ret.stop = str.indexOf(LANG_END, ret.start);
    if (ret.start != -1 && ret.stop != -1) {
      ret.outerStart = start;
      ret.outerStop = ret.stop + LANG_END.length;
      return ret;
    } else {
      return null;
    }
  }

  return null;
}

function replaceCodeInLanguage(lang, str, cb) {
  var start = 0;

  if (!cb) {
    return str;
  }

  while (true) {
    var found = findNextCodeInLanguage(lang, str, start);

    if (found == null) {
      break;
    }

    var replaced = cb(str.substring(found.start, found.stop).replace(/^\s*/, '').replace(/\s*$/, ''));
    str = str.substring(0, found.outerStart) + replaced + str.substring(found.outerStop);
  }

  return str;
}

// function readSync(stream, timeoutInMs) {
//   var buffer = '';
//   var end_flag = false;
//   var startTime = new Date();
//   timeoutInMs = timeoutInMs || 3000;

//   stream.on('data', function (chunk) {
//     buffer += chunk;
//   });

//   stream.on('end', function () {
//     end_flag = true;
//   });

//   while (end_flag != true && new Date() - startTime < timeoutInMs) {
//     // wait
//     process.nextTick();
//   }

//   return stream.read();
// }

// function readStream(stream, cb) {
//   console.log(stream);
//   stream.on('end', function () {
//     cb(null, stream.read());
//   });
// }

// var syncReadStream = deasync(readStream);

function sha256(str) {
  var hash = crypto.createHash('sha256');
  hash.update(str);
  return hash.digest('hex');
}

// headings trigger a new slide
// headings with a caret (e.g., '##^ foo`) trigger a new vertical slide
module.exports = (markdown, options) => {

  var no = 1;

  markdown = replaceCodeInLanguage('puml', markdown, function (original) {

    var mdTitle = (options.title || 'puml').replace(/(\s+|\/|\\)/g, '_').toLowerCase();
    var dir = 'assets/' + mdTitle;
    if (!fs.existsSync(dir)){
        fs.mkdirSync(dir);
    }

    var title = no + '_' + sha256(original);

    var titleMatch = original.match(/^title\s+(.+)$/m);
    if (titleMatch != null && titleMatch[1] != null) {
      title = no + '_' + titleMatch[1].replace(/^"/, '').replace(/\s+$/, '').replace(/"$/, '')
              .replace(/\\"/g, '&quote;').replace(/&quote;/g, '"').replace(/\\\\/g, '\\')
              .replace(/\(/g, '_').replace(/\)/g, '_').toLowerCase();
    }

    var fileName = dir + '/' + title;
    var pumlFileName = fileName + ".puml";
    var svgFileName = fileName + ".svg";

    no += 1;

    var opts = {
      encoding: 'utf-8'
    };
    var lastOriginal = fs.existsSync(pumlFileName) ? fs.readFileSync(pumlFileName, opts) : '';

    if (original != lastOriginal) {
      fs.writeFileSync(pumlFileName, original, opts);

      plantuml.generate(original, {format: 'svg', charset: 'utf-8'}).out.pipe(fs.createWriteStream(svgFileName, opts));
    }

    return '<a href="' + svgFileName + '" data-fancybox="images">![](' + svgFileName + ') <!-- .element class="img-500" --> </a>';

    // return '^_^';
    // return syncReadStream(plantuml.generate(original, {format: 'svg'}).out);
  });

  // deasync.sleep(1000);

  return markdown;
  // return markdown.split('\n').map((line, index) => {
  //   if(!/^#/.test(line) || index === 0) return line;
  //   const is_vertical = /#\^/.test(line);
  //   return (is_vertical ? '+++\n' : '---\n') + line.replace('#^', '#');
  // }).join('\n');
}
