document.getElementById('seekbar').addEventListener('click', function (e) {
    var player = document.getElementById('player');
    var bar = document.getElementById('seekbar');
    var percent = e.offsetX / bar.offsetWidth;
    player.currentTime = percent * player.duration;
});
document.getElementById('player').addEventListener("timeupdate", function() {
	document.getElementById("seekbar").value = this.currentTime/this.duration; 
});
document.getElementById('player').addEventListener("ended", function() {
	document.getElementById("next").click();
});
document.getElementById('player').addEventListener("canplay", function() {
	document.getElementById("currenttime").innerHTML = formatTime(player.currentTime);
	document.getElementById("totallength").innerHTML = formatTime(player.duration);

});
document.getElementById('player').addEventListener("timeupdate", function() {
	document.getElementById("currenttime").innerHTML = formatTime(player.currentTime);

});
function formatTime(seconds) {
    minutes = Math.floor(seconds / 60);
    minutes = (minutes >= 10) ? minutes : "0" + minutes;
    seconds = Math.floor(seconds % 60);
    seconds = (seconds >= 10) ? seconds : "0" + seconds;
    return minutes + ":" + seconds;
}

function playsong(song,file) {
	resetPlayer();
	document.getElementById('songname').innerHTML = song ;
	var player = document.getElementById('player');
	player.src = file;
	player.title = song;
	player.load();
	player.play();
}

function resetPlayer() {
	var playpause = document.getElementById("play");
	var player = document.getElementById('player');
	playpause.innerHTML = "PAUSE";
	if (player.currentTime > 0) player.currentTime = 0;
	
}
function next() {
	var p = document.getElementById('player');
	var title = document.getElementById('songname');
	var currsong = title.innerHTML;
	var pos = 0;
	var ans = -1;
	var nextsongtitle;
	var nextsongurl;
	{% for song in songs %}
		if ('{{ song.song_title }}' == currsong) {
			ans = pos;
		}
		else {
			pos = pos+1;
		}
		if(ans >= 0 && pos == ans+1) {
			nextsongtitle = '{{ song.song_title }}';
			nextsongurl = '{{ song.file.url }}';
		}
	{% endfor %}
	if('{{ songs|length}}' == ans+1) {
		nextsongtitle = '{{ songs.0.song_title }}';
		nextsongurl = '{{ songs.0.file.url }}';
	}
	playsong(nextsongtitle,nextsongurl);
}
function prev() {
	var p = document.getElementById('player');
	var title = document.getElementById('songname');
	var currsong = title.innerHTML;
	var pos = 0;
	var ans = -1;
	var prevsongtitle;
	var prevsongurl;
	var title;
	var url;
	{% for song in songs %}
		if ('{{ song.song_title }}' == currsong) {
			ans = pos;
		}
		else {
			pos = pos+1;
		}
		if(ans >= 0 && pos == ans) {
			prevsongtitle = title;
			prevsongurl = url;
		}
		title = '{{ song.song_title }}';
		url = '{{ song.file.url }}';
	{% endfor %}
	if(ans != 0) {
		playsong(prevsongtitle,prevsongurl);
	}
}
function play() {
	var player = document.getElementById('player');
	var button = document.getElementById('play');
	if(player.paused || player.ended) {
		button.innerHTML = "PAUSE";
		player.play();
	}
	else {
		button.innerHTML = "PLAY";
		player.pause(); 
	}

}

function setVolume() {
   var volume = document.getElementById("volume");
   var player = document.getElementById('player');
   player.volume = volume.value;
}


