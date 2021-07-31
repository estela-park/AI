const widthO = window.outerWidth
const widthI = window.innerWidth

const h1 = document.querySelector('div h1')
const h2 = document.querySelector('h2')

h1.innerText = widthI
h2.innerText = widthO

function handleWindowResize() {
  if (widthO < 500) {
    document.body.style.backgroundColor = "blue";
  } else if (widthO < 700) {
    document.body.style.backgroundColor = "purple";
  } else {
    document.body.style.backgroundColor = "yellow";
  }

  if (widthI < 700) {
    document.body.style.backgroundColor = "blue";
  } else if (widthI < 1100) {
    document.body.style.backgroundColor = "purple";
  } else {
    document.body.style.backgroundColor = "yellow";
  }
}

console.dir(h1)
console.dir(window)
console.dir(document)
console.dir(document.body)

window.addEventListener("resize", handleWindowResize);