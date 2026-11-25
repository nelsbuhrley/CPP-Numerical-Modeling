(function () {
    var themeStorageKey = "site-theme";
    var motionStorageKey = "site-motion";
    var mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
    var navLinks = [
        { title: "Portfolio Hub", href: "/" },
        { title: "Project 1: Realistic Projectile Motion", href: "/Project%201%3A%20realistic%20projectile%20motion/" },
        { title: "Project 2: Driven Damped Oscillations", href: "/Project%202%3A%20driven%20damped%20oscillations/" },
        { title: "Project 3: Celestial Dynamics", href: "/Project%203%3A%20Celestial%20Dynamics/" },
        { title: "Project 4: Overrelaxation", href: "/Project%204%3A%20Overrelaxation/" },
        { title: "Project 5: Oscillations on a String", href: "/Project%205%3A%20Occilations%20on%20a%20string/" },
        { title: "Project 6: Diffusion", href: "/Project%206%3A%20Diffusion/" },
        { title: "Project 7: The Ising Model", href: "/Project%207%3A%20The%20Ising%20Model/" },
        { title: "Project 8: Molecular Dynamics", href: "/Project%208%3A%20Molicular%20Dynamics/" },
        { title: "Project 9: Quantum Mechanics", href: "/Project%209%3A%20Quantum%20Mechanics/" },
        { title: "Personal Project 1: 3n+1", href: "/Personal%20Project%201%3A%203n%2B1/" },
        { title: "Personal Project 2: Idelic Numbers", href: "/Personal%20Project%202%3A%20Idelic%20Numbers/" },
        { title: "Personal Project 3: Monty Hall", href: "/Personal%20Project%203%3A%20Monty%20Hall%20Simulation%20%28Goat%20and%20Car%20Game%20Show%29/" }
    ];

    function getStoredTheme() {
        var value = localStorage.getItem(themeStorageKey);
        if (value === "light" || value === "dark" || value === "auto") {
            return value;
        }
        return "auto";
    }

    function getStoredMotion() {
        var value = localStorage.getItem(motionStorageKey);
        if (value === "default" || value === "reduced") {
            return value;
        }
        return "default";
    }

    function getBaseUrl() {
        var baseUrl = window.__SITE_BASEURL || "";
        if (baseUrl && baseUrl.endsWith("/")) {
            return baseUrl.slice(0, -1);
        }
        return baseUrl;
    }

    function buildPageUrl(relativePath) {
        var baseUrl = getBaseUrl();
        return baseUrl + relativePath;
    }

    function normalizePath(pathname) {
        var baseUrl = getBaseUrl();
        var path = pathname || "/";
        if (baseUrl && path.indexOf(baseUrl) === 0) {
            path = path.slice(baseUrl.length);
        }
        if (!path.endsWith("/")) {
            path += "/";
        }
        return path;
    }

    function getEffectiveTheme(themeChoice) {
        if (themeChoice === "light" || themeChoice === "dark") {
            return themeChoice;
        }
        return mediaQuery.matches ? "dark" : "light";
    }

    function applyTheme(themeChoice) {
        var effectiveTheme = getEffectiveTheme(themeChoice);
        document.documentElement.setAttribute("data-theme", effectiveTheme);
        document.documentElement.style.colorScheme = effectiveTheme;
    }

    function applyMotion(motionChoice) {
        var shouldReduce = motionChoice === "reduced";
        document.documentElement.classList.toggle("reduce-motion", shouldReduce);
    }

    function persistTheme(themeChoice) {
        localStorage.setItem(themeStorageKey, themeChoice);
    }

    function persistMotion(motionChoice) {
        localStorage.setItem(motionStorageKey, motionChoice);
    }

    function buildSelect(options, id, ariaLabel, selectedValue) {
        var select = document.createElement("select");
        select.id = id;
        select.setAttribute("aria-label", ariaLabel);

        options.forEach(function (item) {
            var option = document.createElement("option");
            option.value = item.value;
            option.textContent = item.text;
            if (item.value === selectedValue) {
                option.selected = true;
            }
            select.appendChild(option);
        });

        return select;
    }

    function buildNavList() {
        var nav = document.createElement("nav");
        nav.className = "corner-menu-nav";
        nav.setAttribute("aria-label", "Site navigation");

        var currentPath = normalizePath(window.location.pathname);

        navLinks.forEach(function (item) {
            var link = document.createElement("a");
            var targetPath = normalizePath(item.href);
            link.href = buildPageUrl(item.href);
            link.textContent = item.title;
            if (currentPath === targetPath) {
                link.classList.add("is-active");
            }
            nav.appendChild(link);
        });

        return nav;
    }

    function buildMenu(themeChoice, motionChoice) {
        if (document.querySelector(".corner-menu")) {
            return;
        }

        var container = document.createElement("div");
        container.className = "corner-menu";
        container.setAttribute("data-pinned", "false");

        var toggleButton = document.createElement("button");
        toggleButton.className = "corner-menu-toggle";
        toggleButton.type = "button";
        toggleButton.setAttribute("aria-label", "Open site menu");
        toggleButton.setAttribute("aria-haspopup", "true");
        toggleButton.setAttribute("aria-expanded", "false");
        toggleButton.innerHTML = "<span>Menu</span><span aria-hidden=\"true\">&#9776;</span>";

        var panel = document.createElement("div");
        panel.className = "corner-menu-panel";

        var navSection = document.createElement("section");
        navSection.className = "corner-menu-section corner-menu-section--nav";

        var navTitle = document.createElement("p");
        navTitle.className = "corner-menu-title";
        navTitle.textContent = "Navigate";

        navSection.appendChild(navTitle);
        navSection.appendChild(buildNavList());

        var settingsSection = document.createElement("section");
        settingsSection.className = "corner-menu-section corner-menu-section--settings";

        var settingsTitle = document.createElement("p");
        settingsTitle.className = "corner-menu-title";
        settingsTitle.textContent = "Settings";

        var themeRow = document.createElement("div");
        themeRow.className = "corner-menu-row";

        var themeLabel = document.createElement("label");
        themeLabel.className = "corner-menu-label";
        themeLabel.setAttribute("for", "theme-select");
        themeLabel.textContent = "Theme";

        var themeSelect = buildSelect(
            [
                { value: "auto", text: "Auto" },
                { value: "light", text: "Light" },
                { value: "dark", text: "Dark" }
            ],
            "theme-select",
            "Choose theme",
            themeChoice
        );

        themeSelect.addEventListener("change", function () {
            var nextChoice = themeSelect.value;
            persistTheme(nextChoice);
            applyTheme(nextChoice);
        });

        themeRow.appendChild(themeLabel);
        themeRow.appendChild(themeSelect);

        var motionRow = document.createElement("div");
        motionRow.className = "corner-menu-row";

        var motionLabel = document.createElement("label");
        motionLabel.className = "corner-menu-label";
        motionLabel.setAttribute("for", "motion-select");
        motionLabel.textContent = "Motion";

        var motionSelect = buildSelect(
            [
                { value: "default", text: "Default" },
                { value: "reduced", text: "Reduced" }
            ],
            "motion-select",
            "Choose animation mode",
            motionChoice
        );

        motionSelect.addEventListener("change", function () {
            var nextChoice = motionSelect.value;
            persistMotion(nextChoice);
            applyMotion(nextChoice);
        });

        motionRow.appendChild(motionLabel);
        motionRow.appendChild(motionSelect);

        settingsSection.appendChild(settingsTitle);
        settingsSection.appendChild(themeRow);
        settingsSection.appendChild(motionRow);

        panel.appendChild(settingsSection);
        panel.appendChild(navSection);

        function isPinned() {
            return container.getAttribute("data-pinned") === "true";
        }

        function syncExpandedState() {
            var expanded = isPinned() || container.matches(":hover") || container.matches(":focus-within");
            toggleButton.setAttribute("aria-expanded", String(expanded));
        }

        function setPinned(nextPinned) {
            container.setAttribute("data-pinned", String(nextPinned));
            syncExpandedState();
        }

        toggleButton.addEventListener("click", function () {
            setPinned(!isPinned());
        });

        document.addEventListener("click", function (event) {
            if (!container.contains(event.target)) {
                setPinned(false);
            }
        });

        document.addEventListener("keydown", function (event) {
            if (event.key === "Escape") {
                setPinned(false);
            }
        });

        container.addEventListener("mouseenter", syncExpandedState);
        container.addEventListener("mouseleave", syncExpandedState);
        container.addEventListener("focusin", syncExpandedState);
        container.addEventListener("focusout", function (event) {
            var nextTarget = event.relatedTarget;
            if (nextTarget && container.contains(nextTarget)) {
                return;
            }
            syncExpandedState();
        });

        panel.addEventListener("click", function (event) {
            if (event.target && event.target.closest("a")) {
                setPinned(false);
            }
        });

        container.appendChild(toggleButton);
        container.appendChild(panel);
        document.body.appendChild(container);
        syncExpandedState();
    }


    function addSiteLinks() {
        var mainSite = "https://nels.buhrley.com/";
        var base = getBaseUrl() + "/";
        var path = window.location.pathname.replace(/index\.html$/, "");
        var header = document.querySelector(".page-header");

        if (header && path.indexOf(base) === 0 && !header.querySelector(".site-breadcrumb")) {
            var items = '<li><a href="' + mainSite + '">Home</a></li>';
            if (path === base) {
                items += '<li aria-current="page">Computational Physics Portfolio</li>';
            } else {
                var title = "";
                navLinks.forEach(function (item) {
                    if (item.href !== "/" && decodeURIComponent(path).indexOf(decodeURIComponent(buildPageUrl(item.href))) === 0) {
                        title = item.title;
                    }
                });
                if (!title) {
                    var h1 = document.querySelector(".main-content h1");
                    title = h1 ? h1.textContent.trim() : document.title;
                }
                var safe = document.createElement("span");
                safe.textContent = title;
                items += '<li><a href="' + base + '">Computational Physics Portfolio</a></li>' +
                    '<li aria-current="page">' + safe.innerHTML + '</li>';
            }
            var crumbs = document.createElement("nav");
            crumbs.className = "site-breadcrumb";
            crumbs.setAttribute("aria-label", "Breadcrumb");
            crumbs.innerHTML = "<ol>" + items + "</ol>";
            header.insertBefore(crumbs, header.firstChild);
        }

        var footer = document.querySelector(".site-footer");
        if (footer && !footer.querySelector(".site-footer-home")) {
            var back = document.createElement("p");
            back.className = "site-footer-home";
            back.innerHTML = '<a href="' + mainSite + '">&larr; Back to nels.buhrley.com</a>';
            footer.appendChild(back);
        }
    }

    function initializeMenu() {
        var themeChoice = getStoredTheme();
        var motionChoice = getStoredMotion();

        applyTheme(themeChoice);
        applyMotion(motionChoice);

        if (themeChoice === "auto") {
            mediaQuery.addEventListener("change", function () {
                applyTheme("auto");
            });
        }

        if (document.readyState === "loading") {
            document.addEventListener("DOMContentLoaded", function () {
                buildMenu(themeChoice, motionChoice);
                addSiteLinks();
            });
        } else {
            buildMenu(themeChoice, motionChoice);
            addSiteLinks();
        }
    }

    initializeMenu();
})();
