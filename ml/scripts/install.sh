export PUPPETEER_SKIP_CHROMIUM_DOWNLOAD=1
npm install -g reveal-md
npm install -g mathjax-node-cli

# Intall graphviz
# https://graphviz.gitlab.io/_pages/Download/Download_windows.html
# http://www.graphviz.org/Download..php
# brew install graphviz
sudo apt update
sudo apt install -y graphviz

# http://ogom.github.io/draw_uml/plantuml/
npm install -g node-plantuml
npm link node-plantuml

# # Add `npm root -g` to NODE_PATH
# export NODE_PATH=$NODE_PATH:`npm root -g`