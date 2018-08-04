How to view this slide
=============================

Online
----------

Just visit https://gitpitch.com/utensil/slides/master?p=ml , but some slides can't be shown due to origin restrictions and some functionality is absent due to GitPitch restrictions.

Local
----------

1. install latest [node.js](https://nodejs.org/en/)
2. open a console, run `set PUPPETEER_SKIP_CHROMIUM_DOWNLOAD=1` for Windows and `export PUPPETEER_SKIP_CHROMIUM_DOWNLOAD=1
` for Linux/Mac
3. run `npm install -g reveal-md` to install reveal-md
4. run `git clone https://github.com/utensil/slides` and `cd slides/ml`
5. run `reveal-md -w ml.md` to start serving the slide, the browser should automatically open the slide, if not, visit http://localhost:1948 .
