$(document).ready(function() {
    $(".math").each(function(i) { $(this).html("<img src=\"https://latex.codecogs.com/svg.latex?" + this.innerHTML + "\"/>"); });
});
