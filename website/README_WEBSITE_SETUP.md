# Website setup

1. Copy the contents of this `website` folder into the existing `website/` folder in your project.
2. In `index.html`, replace every `YOUR-USERNAME` and `YOUR-REPOSITORY` with your actual GitHub username and repository name.
3. Open the project folder in VS Code.
4. Install/enable the Live Server extension if you do not already have it.
5. Right-click `website/index.html` and choose **Open with Live Server**.
6. Review every number and claim against the latest executed notebook outputs before publishing.
7. When ready, push the project to GitHub. In repository Settings → Pages, choose **Deploy from a branch**, select `main`, and select `/website` as the publishing folder if GitHub offers it. If the publishing-source dropdown only offers `/docs` or root, rename `website` to `docs` or use a GitHub Actions workflow. Do not rename until you decide the publishing setup.

The charts use Chart.js from a CDN, so the browser needs an internet connection to load the chart library.
