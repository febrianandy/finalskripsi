$(function(){
	$('#svm').click(function(){
		var hate_speech = $('#hate_speech').val();
		$.ajax({
			url: '/svm',
			data: $('form').serialize(),
			type: 'POST',
			beforeSend: function(){
        $('.loading').addClass('loading-show');
			
			},
			success: function(response){
        $('.algoritma').text('Support Vector Machine');
        $('.result').text(response.hate_speech);
        $('.accuracy').text(response.accuracy);
			},
			error: function(error){
				console.log(error);
			},
			complete: function(){
				$('.loading').removeClass('loading-show');
		
			},
		});
	});

  $('#tree').click(function(){
		var hate_speech_input = $('#hate_speech').val();
		$.ajax({
			url: '/decisiontree',
			data: $('form').serialize(),
			type: 'POST',
			beforeSend: function(){
        $('.loading').addClass('loading-show');
			},
			success: function(response){
        $('.algoritma').text('Decision Tree');
        $('.result').text(response.hate_speech);
        $('.accuracy').text(response.accuracy);
			},
			error: function(error){
				console.log(error);
			},
			complete: function(){
				$('.loading').removeClass('loading-show');
			},
		});
	});

	$('#bagging').click(function(){
		var hate_speech_input = $('#hate_speech').val();
		$.ajax({
			url: '/bagging',
			data: $('form').serialize(),
			type: 'POST',
			beforeSend: function(){
				$('.loading').addClass('loading-show');
			},
			success: function(response){
				$('.algoritma').text('Bagging Classifier Decision Tree');
				$('.result').text(response.hate_speech);
				$('.accuracy').text(response.accuracy);
			},
			error: function(error){
				console.log(error);
			},
			complete: function(){
				$('.loading').removeClass('loading-show');
			},
		});
	});
})