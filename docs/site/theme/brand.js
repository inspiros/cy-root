/* Progressive branding; mdBook owns navigation, search, and reader preferences. */
(() => {
    const bar = document.querySelector('.menu-bar .left-buttons');
    if (!bar) return;
    const home = document.createElement('a');
    home.className = 'cyroot-menu-logo';
    home.href = 'index.html';
    home.setAttribute('aria-label', 'CyRoot documentation home');
    const mark = document.createElement('img');
    mark.src = 'assets/cyroot_logo.svg';
    mark.alt = '';
    const name = document.createElement('span');
    name.textContent = 'CyRoot';
    const release = document.createElement('span');
    release.className = 'release';
    release.textContent = '2.0';
    home.append(mark, name, release);
    bar.append(home);
    const refreshLogo = () => {
        const dark = ['navy', 'coal', 'ayu'].some(theme => document.documentElement.classList.contains(theme));
        mark.src = dark ? 'assets/cyroot_logo-dark.svg' : 'assets/cyroot_logo.svg';
    };
    refreshLogo();
    new MutationObserver(refreshLogo).observe(document.documentElement, {attributes: true, attributeFilter: ['class']});
})();
