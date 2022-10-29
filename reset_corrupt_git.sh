rm -fr .git
git init
git remote add origin https://github.com/nkasmanoff/nyc-sentinel
git fetch
git reset --mixed origin/main
git checkout -b main
git branch -D master
git branch --set-upstream-to=origin/main maindf