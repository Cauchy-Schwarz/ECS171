$("#formElem").submit(function(e) {
    e.preventDefault();    
    var formData = new FormData(this);

    $.ajax({
        url: "/predict",
        type: 'POST',
        data: formData,
        success: function (data) {
            let result = document.getElementById("predicted_model");
            result.innerHTML = data;
            result.style.display = "block";
        },
        cache: false,
        contentType: false,
        processData: false
    });
});