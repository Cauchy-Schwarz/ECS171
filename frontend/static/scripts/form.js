$("#formElem").submit(function(e) {
    e.preventDefault();    
    var formData = new FormData(this);

    $.ajax({
        url: "/predict",
        type: 'POST',
        data: formData,
        success: function (data) {
            let result1 = document.getElementById("predicted_model1");
            let result2 = document.getElementById("predicted_model2");
            result1.innerHTML = "effnetLArch: " + data[0];
            result1.style.display = "block";
            result2.innerHTML = "effnetMArch: " + data[1];
            result2.style.display = "block";
        },
        cache: false,
        contentType: false,
        processData: false
    });
});