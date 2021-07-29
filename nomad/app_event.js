const title = document.querySelector('#title')
const h1Tag = document.querySelector('div h1')
console.dir(title)

function handleTitleClick() {
    title.innerText = 'My name is Stella'
}
function handleMouseEnter() {
    h1Tag.innerText = 'Pointer is here'
}
function handleClickColorChange() {
    h1Tag.style.color = 'magenta'
}
function handleMouseLeave() {
    h1Tag.innerText = 'Pointer is leaving'
}
 function handleWindowResize() {
    document.body.style.backgroundColor = 'lavender'
}
function handleWindowCopy() {
    alert("Don't copy!")
}
function handleWindowOffline() {
    alert("no WIFI orz")
}
function handleWindowOnline() {
    alert("Welcome :)")
}


title.addEventListener('click', handleTitleClick)
h1Tag.addEventListener('click', handleClickColorChange)
h1Tag.addEventListener('mouseenter', handleMouseEnter)
h1Tag.addEventListener('mouseleave', handleMouseLeave)
window.addEventListener('resize', handleWindowResize)
window.addEventListener('copy', handleWindowCopy)
window.addEventListener('offline', handleWindowOffline)