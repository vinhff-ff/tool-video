// animations/images.js
// Controls visibility of the A/B comparison images.
const ImagesCtl = {
  imgA: null,
  imgB: null,

  init(imgA, imgB) {
    this.imgA = imgA;
    this.imgB = imgB;
  },

  showA() {
    this.imgA.classList.add("img--active");
    this.imgB.classList.remove("img--active");
  },

  showB() {
    this.imgB.classList.add("img--active");
    this.imgA.classList.remove("img--active");
  },

  showBoth() {
    this.imgA.classList.add("img--active");
    this.imgB.classList.add("img--active");
  },

  hideAll() {
    this.imgA.classList.remove("img--active");
    this.imgB.classList.remove("img--active");
  },
};