// Function from uploading picture
// Displays image
function showPreview(event) {
  if (event.target.files.length > 0) {
    let src = URL.createObjectURL(event.target.files[0]);
    let input = document.getElementById("file-ip-1");
    let preview = document.getElementById("file-preview");
    let info = document.getElementById("file-info");
    let submitBtn = document.getElementById("submit_btn");
    let result1 = document.getElementById("predicted_model1");
    let result2 = document.getElementById("predicted_model2");
    result1.style.display = "none";
    result2.style.display = "none";
    testingPicture = src;
    pictureType = input.files.item(0).type;
    preview.src = src;
    preview.style.display = "block";
    preview.style.width = "80%";
    preview.style.height = "80%";
    info.innerHTML = input.files.item(0).name;
    info.style.display = "block";
    submitBtn.style.display = "block";
  }
}

