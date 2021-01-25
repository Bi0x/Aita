// Navigaton Animation
function navigationMouseOn() {
    document.getElementById("navigationSelectedItem").style.display = "none";
    document.getElementById("navigationSelectedItemAnimate").style.display = "inline";
}

function navigationMouseOut() {
    document.getElementById("navigationSelectedItemAnimate").style.display = "none";
    document.getElementById("navigationSelectedItem").style.animation = "navigationFadeIn .3s"
    document.getElementById("navigationSelectedItem").style.display = "inline";
}
// Navigaton Animation