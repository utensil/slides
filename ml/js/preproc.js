// headings trigger a new slide
// headings with a caret (e.g., '##^ foo`) trigger a new vertical slide
module.exports = (markdown, options) => {
  return markdown;
  // return markdown.split('\n').map((line, index) => {
  //   if(!/^#/.test(line) || index === 0) return line;
  //   const is_vertical = /#\^/.test(line);
  //   return (is_vertical ? '+++\n' : '---\n') + line.replace('#^', '#');
  // }).join('\n');
};