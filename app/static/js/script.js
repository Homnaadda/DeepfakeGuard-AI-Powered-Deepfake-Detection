$(document).ready(function() {
    // Helper to format file sizes
    function formatBytes(bytes, decimals = 2) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
    }

    // --- STICKY NAVBAR SCROLL ACTION ---
    const $navbar = $('#main-navbar');
    function checkNavbarScroll() {
        if ($(window).scrollTop() > 30) {
            $navbar.addClass('scrolled');
        } else {
            $navbar.removeClass('scrolled');
        }
    }
    checkNavbarScroll();
    $(window).on('scroll', checkNavbarScroll);

    // --- LIQUID GLASS SPOTLIGHT HOVER ---
    const $spotlight = $('#nav-spotlight');
    const $navList = $('#nav-list');
    const $navLinks = $('#nav-list .nav-link');

    function moveSpotlight($item) {
        if (!$item.length || !$item.is(':visible')) {
            $spotlight.css('opacity', 0);
            return;
        }
        const itemPos = $item.position();
        
        $spotlight.css({
            width: $item.outerWidth(),
            height: $item.outerHeight(),
            left: itemPos.left,
            top: itemPos.top,
            opacity: 1
        });
    }

    $navLinks.on('mouseenter', function() {
        moveSpotlight($(this));
    });

    $navList.on('mouseleave', function() {
        const $active = $navLinks.filter('.active');
        if ($active.length) {
            moveSpotlight($active);
        } else {
            $spotlight.css('opacity', 0);
        }
    });

    // Handle initial load spotlight positioning
    setTimeout(function() {
        const $active = $navLinks.filter('.active');
        if ($active.length) moveSpotlight($active);
    }, 250);


    // --- DYNAMIC SCROLL ACTIVE NAV LINK HIGHLIGHT & SCROLLSPY ---
    function updateActiveNav() {
        const path = window.location.pathname;
        
        // If on Detection Center (/detection) or dedicated detection/results subpages, highlight "Detection"
        if (path.includes('/detection') || path.includes('-detection') || path.includes('/predict')) {
            $navLinks.removeClass('active');
            $navLinks.filter('[href*="/detection"]').addClass('active');
            
            // Move spotlight to active element
            const $active = $navLinks.filter('.active');
            if ($active.length) moveSpotlight($active);
            return;
        }

        // Only run scrollspy if we are on the homepage (contains #hero)
        if ($('#hero').length === 0) return;

        const scrollPos = $(window).scrollTop() + 150;
        const sections = [
            { id: 'hero', link: '/#hero' },
            { id: 'models', link: '/#models' },
            { id: 'project-explanation', link: '/#project-explanation' },
            { id: 'contact', link: '/#contact' }
        ];

        let activeIndex = 0;
        for (let i = 0; i < sections.length; i++) {
            const $sec = $('#' + sections[i].id);
            if ($sec.length && scrollPos >= $sec.offset().top) {
                activeIndex = i;
            }
        }

        $navLinks.removeClass('active');
        const targetHref = sections[activeIndex].link;
        const $targetLink = $navLinks.filter(`[href="${targetHref}"]`);
        $targetLink.addClass('active');

        // Reposition spotlight on active nav
        const $active = $navLinks.filter('.active');
        if ($active.length) moveSpotlight($active);
    }

    $(window).on('scroll', updateActiveNav);
    updateActiveNav();


    // --- SMOOTH ANCHOR SCROLLING ---
    $('a.nav-scroll-link').on('click', function(e) {
        const hash = this.hash;
        if (hash) {
            // Check if element exists on current page
            const $target = $(hash);
            if ($target.length) {
                e.preventDefault();
                $('html, body').stop().animate({
                    scrollTop: $target.offset().top - 80
                }, 800);
            }
        }
    });


    // --- DEDICATED PAGES UPLOAD HANDLERS ---
    // Handle image, video, and audio file uploads separately on their dedicated templates

    // Universal file handler helper
    function handleFileSelection(file, type) {
        if (!file) return;

        const $details = $(`#${type}-file-details`);
        $details.find('.file-name').text(file.name);
        $details.find('.file-size').text(formatBytes(file.size));
        $details.show();
        $(`#drag-drop-${type}`).addClass('has-file');

        // Specific previews per type
        if (type === 'image') {
            const reader = new FileReader();
            reader.onload = function(e) {
                $('#image-preview-img').attr('src', e.target.result);
                $('#image-preview-box').fadeIn(300);
            };
            reader.readAsDataURL(file);
        } else if (type === 'video') {
            const videoUrl = URL.createObjectURL(file);
            $('#video_source').attr('src', videoUrl);
            const $videoPlayer = $('#videos');
            $videoPlayer.show();
            $videoPlayer[0].load();
        }
    }

    // Set up events for the three types
    ['image', 'video', 'audio'].forEach(function(type) {
        const $dropzone = $(`#drag-drop-${type}`);
        const $input = $(`#id_upload_${type}_file`);

        if ($dropzone.length && $input.length) {
            // Bind input change
            $input.on('change', function() {
                if (this.files && this.files[0]) {
                    handleFileSelection(this.files[0], type);
                }
            });

            // Bind drag-drop zone interactions
            $dropzone.on('dragover dragenter', function(e) {
                e.preventDefault();
                e.stopPropagation();
                $dropzone.addClass('dragover');
            });

            $dropzone.on('dragleave drop', function(e) {
                e.preventDefault();
                e.stopPropagation();
                $dropzone.removeClass('dragover');
            });

            $dropzone.on('drop', function(e) {
                const files = e.originalEvent.dataTransfer.files;
                if (files && files.length > 0) {
                    $input[0].files = files;
                    handleFileSelection(files[0], type);
                }
            });
        }
    });


    // --- FORENSIC SCANNING INTERACTIVE CONSOLE LOADER ---
    const loadingMessages = [
        "Initializing forensic analyzer...",
        "Uploading media data to detection nodes...",
        "Aligning spatial features & processing frames...",
        "Analyzing structural inconsistencies...",
        "Aggregating classifier scores & compiling report...",
        "Finalizing diagnostic parameters..."
    ];

    let consoleInterval = null;

    function startLoaderConsole(messages, callback = null) {
        $('#loader-overlay').css('display', 'flex');
        let index = 0;
        $('#loader-status-text').text(messages[index]);
        
        if (consoleInterval) clearInterval(consoleInterval);

        consoleInterval = setInterval(function() {
            index++;
            if (index < messages.length) {
                $('#loader-status-text').fadeOut(200, function() {
                    $(this).text(messages[index]).fadeIn(200);
                });
            } else {
                clearInterval(consoleInterval);
                if (callback) {
                    callback();
                }
            }
        }, 1500);
    }

    // Attach loader submit events
    const $imageForm = $('#image-upload');
    const $videoForm = $('#video-upload');
    const $audioForm = $('#audio-upload');

    // Image Form Submit
    if ($imageForm.length) {
        $imageForm.on('submit', function() {
            const $submitBtn = $imageForm.find('button[type="submit"]');
            $submitBtn.prop("disabled", true);
            startLoaderConsole(loadingMessages);
        });
    }

    // Video Form Submit
    if ($videoForm.length) {
        $videoForm.on('submit', function() {
            const $submitBtn = $videoForm.find('button[type="submit"]');
            $submitBtn.prop("disabled", true);
            startLoaderConsole(loadingMessages);
        });
    }

    // Audio Form Submit
    if ($audioForm.length) {
        $audioForm.on('submit', function() {
            const $submitBtn = $audioForm.find('button[type="submit"]');
            $submitBtn.prop("disabled", true);
            startLoaderConsole(loadingMessages);
        });
    }


});