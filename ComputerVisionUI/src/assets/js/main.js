	//toggle navigation
	//$("body").toggleClass("sidenav-toggled");

	function myFunction() {
	    console.log('my function called');
	    $("body").toggleClass("sidenav-toggled");
	    $(".navbar-sidenav .nav-link-collapse").removeClass("collapsed");
	    $(".navbar-sidenav .sidenav-second-level, .navbar-sidenav .sidenav-third-level").removeClass("show");
	}
	// var slider = document.getElementById("myRange");
	// var output = document.getElementById("demo");
	// output.innerHTML = slider.value;

	// slider.oninput = function() {
	//   output.innerHTML = this.value;
	// }

	(function($) {
	    "use strict"; // Start of use strict 
	    // Toggle the side navigation

	    $("#sidenavToggler").on('click', function(e) {
	        e.preventDefault();
	        $("body").toggleClass("sidenav-toggled");

	        $(".navbar-sidenav .nav-link-collapse").addClass("collapsed");
	        $(".navbar-sidenav .sidenav-second-level, .navbar-sidenav .sidenav-third-level").removeClass("show");
	    });
	    // Force the toggled class to be removed when a collapsible nav link is clicked
	    $(".navbar-sidenav .nav-link-collapse").on('click', function(e) {
	        e.preventDefault();
	        $("body").removeClass("sidenav-toggled");
	    });
	    // Prevent the content wrapper from scrolling when the fixed side navigation hovered over
	    $('body.fixed-nav .navbar-sidenav, body.fixed-nav .sidenav-toggler, body.fixed-nav .navbar-collapse').on('mousewheel DOMMouseScroll', function(e) {
	        var e0 = e.originalEvent,
	            delta = e0.wheelDelta || -e0.detail;
	        this.scrollTop += (delta < 0 ? 1 : -1) * 30;
	        e.preventDefault();
	    });
	    // Scroll to top button appear
	    $(document).scroll(function() {
	        var scrollDistance = $(this).scrollTop();
	        if (scrollDistance > 100) {
	            $('.scroll-to-top').fadeIn();
	        } else {
	            $('.scroll-to-top').fadeOut();
	        }
	    });
	    // Configure tooltips globally
	    $('[data-toggle="tooltip"]').tooltip()
	        // Smooth scrolling using jQuery easing
	    $(document).on('click', 'a.scroll-to-top', function(event) {
	        var $anchor = $(this);
	        $('html, body').stop().animate({
	            scrollTop: ($($anchor.attr('href')).offset().top)
	        }, 1000, 'easeInOutExpo');
	        event.preventDefault();
	    });
	})(jQuery); // End of use strict2
	$(document).on('click', '.next', function() {
	    StepToNext();

	});
	$(document).on('click', '.prev', function() {
	    StepToPrev();
	});
	$(document).on('click', '.finished', function() {
	    //StepToInitial();
	});

	function StepToNext() {
	    if (($('.process-item.is-current').next('.process-item')).length) {
	        $('.process-item.is-current').addClass("is-changing");


	        $('.process-item.is-current.is-changing').removeClass('is-current').addClass('is-active');

	        setTimeout(function() {
	            $('.process-item.is-changing').next('.process-item').addClass("is-current");
	            $('.process-item.is-current').prev('.process-item.is-changing').removeClass('is-changing');
	        }, 250)


	    } else {
	        var itemCount;
	        itemCount = $('.process-item').length
	        console.log(itemCount);
	        $('.process-item.is-current').addClass('is-active').removeClass('is-current');
	        $('.process-item').addClass('all-complete');

	        $('.next').addClass('is-slidedown').removeClass('is-slideup');
	        $('.prev').addClass('is-slidedown').removeClass('is-slideup');


	        setTimeout(function() {
	            $('.next').addClass('is-hide').removeClass('is-show');
	            $('.prev').addClass('is-hide').removeClass('is-show');
	            $('.finished').addClass('is-show').removeClass('is-hide');
	            $('.finished').addClass('is-slideup').removeClass('is-slidedown');
	            $('.complete-hint').addClass('is-show').removeClass('is-hide');
	        }, 120)

	        setTimeout(function() {
	            $('.next').removeClass('is-slidedown').removeClass('is-slideup');
	            $('.prev').removeClass('is-slidedown').removeClass('is-slideup');

	            $('.finished').removeClass('is-slidedown').removeClass('is-slideup');

	        }, 240);

	    }
	    $('.star .radiance').addClass('is-active');
	}

	function StepToPrev() {

	    if (($('.process-item.is-current').prev('.process-item')).length) {

	        $('.process-item.is-current').prev('.process-item').addClass("is-changing")

	        $('.process-item.is-current').removeClass("is-current");

	        $('.process-item.is-changing').addClass('is-current').removeClass('is-active').removeClass('is-changing');
	    } else {
	        return;
	    }

	}

	function StepToInitial() {

	    $('.process-item').removeClass("is-current").removeClass("is-active").removeClass("all-complete");
	    $('.process-item:first-child').addClass("is-current");

	    $('.complete-hint').addClass('is-hide').removeClass('is-show');

	    $('.finished').removeClass('is-slideup').addClass('is-slidedown');


	    setTimeout(function() {
	        $('.next').addClass('is-show').removeClass('is-hide');
	        $('.prev').addClass('is-show').removeClass('is-hide');

	        $('.next').removeClass('is-slidedown').addClass('is-slideup');
	        $('.prev').removeClass('is-slidedown').addClass('is-slideup');

	        $('.finished').addClass('is-hide').removeClass('is-show');

	    }, 120)

	    setTimeout(function() {
	        $('.next').removeClass('is-slidedown').removeClass('is-slideup');
	        $('.prev').removeClass('is-slidedown').removeClass('is-slideup');

	        $('.finished').removeClass('is-slidedown').removeClass('is-slideup');
	    }, 240);

	}