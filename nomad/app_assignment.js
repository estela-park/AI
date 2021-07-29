const colors = ["red", "yellow", "blue", "magenta", "lavender"];
const title = document.querySelector("#title");

function handleMouseEnter() {
  title.innerText = "The pointer is here";
  title.style.color = colors[0];
}

function handleMouseLeaving() {
  title.innerText = "The pointer is leaving";
  title.style.color = colors[2];
}

function handleContextMenu() {
  title.innerText = "This has been clicked with right button";
  title.style.color = colors[3];
}

function handleWindowResize() {
  title.innerText = "The size of window is adjusted";
  title.style.color = colors[4];
}

title.addEventListener("mouseenter", handleMouseEnter);
title.addEventListener("mouseleave", handleMouseLeaving);
window.addEventListener("resize", handleWindowResize);
document.addEventListener("contextmenu", handleContextMenu);
