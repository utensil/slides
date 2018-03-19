/*
> findNextCodeInLanguage('lang', 'abc```lang xxxxxx ```asdf```lang ijbieiuh ```sxdede')
{ start: 10, stop: 18 }
> 'abc```lang xxxxxx ```asdf```lang ijbieiuh ```sxdede'.substring(10, 18)
' xxxxxx '

> findNextCodeInLanguage('lang', 'abc```lang xxxxxx ```asdf```lang ijbieiuh ```sxdede', 18)
{ start: 32, stop: 42 }
> 'abc```lang xxxxxx ```asdf```lang ijbieiuh ```sxdede'.substring(32, 42)
' ijbieiuh '
 */
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
