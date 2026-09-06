// animations/character.js
// Controls characters: main (center / point left / point right / point left-up /
// / point right), plus optional extra characters ("confused", "cart") that are
// swapped in per scene. Only the character named in the scene is visible.
const Character = {
  els: { main: null, confused: null, cart: null },

  init(mainEl, confusedEl, cartEl) {
    this.els.main = mainEl;
    this.els.confused = confusedEl;
    this.els.cart = cartEl;
    this.show("main");
  },

  // Show one character (hide the rest). name in ["main", "confused", "cart"].
  show(name) {
    name = name || "main";
    for (const key in this.els) {
      const el = this.els[key];
      if (el) el.classList.toggle("char--active", key === name);
    }
  },

  center() {
    this.show("main");
    this.els.main.classList.remove("char--left", "char--right", "char--up");
  },

  pointLeft() {
    this.show("main");
    this.els.main.classList.remove("char--right", "char--up");
    this.els.main.classList.add("char--left");
  },

  // Character flips to face the left (no body rotation) - used for the
  // opening line pointing at object A on the top-left.
  pointLeftUp() {
    this.show("main");
    this.els.main.classList.remove("char--right");
    this.els.main.classList.add("char--left");
  },

  pointRight() {
    this.show("main");
    this.els.main.classList.remove("char--left", "char--up");
    this.els.main.classList.add("char--right");
  },
};