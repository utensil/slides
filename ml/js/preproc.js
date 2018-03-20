function testfindNextCodeInLanguage() {
  var lang = 'lang';
  var md = 'abc```lang\nxxxxxx\n```asdf```lang\nijbieiuh\n```sxdede';
  var found = findNextCodeInLanguage(lang, md);
  console.log(md.substring(found.start, found.stop));
  console.log(md.substring(found.outerStart, found.outerStop));

  found = findNextCodeInLanguage(lang, md, found.outerStop);
  console.log(md.substring(found.start, found.stop));
  console.log(md.substring(found.outerStart, found.outerStop));
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

// headings trigger a new slide
// headings with a caret (e.g., '##^ foo`) trigger a new vertical slide
module.exports = (markdown, options) => {
  return markdown;
  // return markdown.split('\n').map((line, index) => {
  //   if(!/^#/.test(line) || index === 0) return line;
  //   const is_vertical = /#\^/.test(line);
  //   return (is_vertical ? '+++\n' : '---\n') + line.replace('#^', '#');
  // }).join('\n');
}

testfindNextCodeInLanguage();