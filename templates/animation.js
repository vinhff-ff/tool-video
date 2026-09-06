// templates/animation.js
// Fixed animation library. The Designer AI may only reference these names
// in Scene JSON ("animation": "<name>") - it never writes new animation code.
const AnimationLibrary = {
  intro(ctx)   { Character.center(); ImagesCtl.hideAll(); ctx.stage.classList.remove("fade-out"); },
  outro(ctx)   { Character.center(); ctx.stage.classList.remove("fade-out"); }, // no dark fade at the end
  fadeIn(ctx)  { ctx.stage.classList.remove("fade-out"); ctx.stage.classList.add("fade-in"); },
  fadeOut(ctx) { ctx.stage.classList.remove("fade-in"); ctx.stage.classList.add("fade-out"); },
  slideLeft(ctx)  { ctx.stage.classList.add("slide-left"); },
  slideRight(ctx) { ctx.stage.classList.add("slide-right"); },
  zoomIn(ctx)  { ctx.stage.classList.add("zoom-in"); },
  zoomOut(ctx) { ctx.stage.classList.add("zoom-out"); },
  showA(ctx)   { ImagesCtl.showA(); },
  showB(ctx)   { ImagesCtl.showB(); },
  compare(ctx) { ImagesCtl.showBoth(); },
  showConfused(ctx) { Character.show("confused"); },
  showCart(ctx)     { Character.show("cart"); },
};

function getActiveCharacter(name) {
  // "pointLeftUp", "pointLeft", "pointRight", "center" → main character with a pose
  // "confused" / "cart" → dedicated extra characters
  if (name === "confused" || name === "cart") return name;
  return "main";
}

async function playScene(sceneData) {
  const stage = document.getElementById("stage");
  const charEl = document.getElementById("character");
  const confusedEl = document.getElementById("character-confused");
  const cartEl = document.getElementById("character-cart");
  const imgA = document.getElementById("image-a");
  const imgB = document.getElementById("image-b");
  const textEl = document.getElementById("caption");

  Character.init(charEl, confusedEl, cartEl);
  ImagesCtl.init(imgA, imgB);
  TextCtl.init(textEl);

  const ctx = { stage };

  for (const scene of sceneData.scenes) {
    // Named animation trigger (visual effect on the stage)
    const anim = AnimationLibrary[scene.animation];
    if (anim) {
      anim(ctx);
    } else if (scene.animation) {
      console.warn("Unknown animation name:", scene.animation);
    }

    // Which image(s) to show this scene
    if (scene.image === "A") ImagesCtl.showA();
    else if (scene.image === "B") ImagesCtl.showB();
    else if (scene.image === "both") ImagesCtl.showBoth();
    else ImagesCtl.hideAll();

    // Character pose this scene (main character + pose, or extra character)
    Character.show(getActiveCharacter(scene.character));
    if (scene.character === "pointLeft") Character.pointLeft();
    else if (scene.character === "pointLeftUp") Character.pointLeftUp();
    else if (scene.character === "pointRight") Character.pointRight();
    else if (scene.character === "center") Character.center();

    // Caption text this scene
    TextCtl.setText(scene.text);

    const durationMs = (scene.duration || 4) * 1000;
    await new Promise((resolve) => setTimeout(resolve, durationMs));
  }

  // Signals recorder.py that playback has finished and it's safe to stop recording.
  window.__SCENE_DONE__ = true;
}

// SCENE_DATA is injected as a literal by renderer.py before this script runs.
playScene(SCENE_DATA);