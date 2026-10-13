const API_URL = "http://127.0.0.1:8000/predict";


// ==========================================
// GET ELEMENTS
// ==========================================

const reviewInput =
    document.getElementById("review");

const analyzeBtn =
    document.getElementById("analyzeBtn");

const clearBtn =
    document.getElementById("clearBtn");

const result =
    document.getElementById("result");

const placeholder =
    document.getElementById("placeholder");

const error =
    document.getElementById("error");

const stars =
    document.getElementById("stars");

const sentiment =
    document.getElementById("sentiment");

const confidenceValue =
    document.getElementById("confidenceValue");

const progressBar =
    document.getElementById("progressBar");


// ==========================================
// ANALYZE REVIEW
// ==========================================

async function analyzeReview() {

    const review =
        reviewInput.value.trim();


    // Empty review

    if (!review) {

        showError(
            "Please enter a product review."
        );

        return;
    }


    // Hide previous error

    error.style.display = "none";


    // Loading state

    analyzeBtn.disabled = true;

    analyzeBtn.textContent =
        "Analyzing...";


    try {

        const response =
            await fetch(
                API_URL,
                {
                    method: "POST",

                    headers: {
                        "Content-Type": "application/json"
                    },

                    body: JSON.stringify({
                        review: review
                    })
                }
            );


        const data =
            await response.json();


        if (!response.ok) {

            throw new Error(
                data.detail ||
                "Prediction failed."
            );

        }


        // ======================================
        // RATING
        // ======================================

        stars.textContent =
            "⭐".repeat(data.rating);


        // ======================================
        // SENTIMENT
        // ======================================

        sentiment.textContent =
            data.sentiment;


        // Remove previous sentiment classes

        sentiment.className =
            "sentiment";


        const sentimentText =
            data.sentiment.toLowerCase();


        if (
            sentimentText.includes(
                "very positive"
            )
        ) {

            sentiment.classList.add(
                "very-positive"
            );

        }

        else if (
            sentimentText.includes(
                "positive"
            )
        ) {

            sentiment.classList.add(
                "positive"
            );

        }

        else if (
            sentimentText.includes(
                "very negative"
            )
        ) {

            sentiment.classList.add(
                "very-negative"
            );

        }

        else if (
            sentimentText.includes(
                "negative"
            )
        ) {

            sentiment.classList.add(
                "negative"
            );

        }

        else {

            sentiment.classList.add(
                "neutral"
            );

        }


        // ======================================
        // CONFIDENCE
        // ======================================

        const confidence =
            (
                data.confidence * 100
            ).toFixed(2);


        confidenceValue.textContent =
            confidence + "%";


        progressBar.style.width =
            confidence + "%";


        // ======================================
        // SHOW RESULT
        // ======================================

        placeholder.style.display =
            "none";

        result.style.display =
            "block";


    }

    catch (err) {

        console.error(err);

        showError(
            "Unable to connect to the API. Make sure FastAPI is running."
        );

    }

    finally {

        analyzeBtn.disabled = false;

        analyzeBtn.textContent =
            "Analyze Review";

    }

}


// ==========================================
// CLEAR REVIEW
// ==========================================

function clearReview() {

    reviewInput.value = "";

    result.style.display =
        "none";

    placeholder.style.display =
        "flex";

    error.style.display =
        "none";

    stars.textContent = "";

    sentiment.textContent = "";

    confidenceValue.textContent =
        "0%";

    progressBar.style.width =
        "0%";

}


// ==========================================
// SHOW ERROR
// ==========================================

function showError(message) {

    error.textContent =
        message;

    error.style.display =
        "block";

}


// ==========================================
// EXAMPLE BUTTONS
// ==========================================

const exampleButtons =
    document.querySelectorAll(
        ".example-btn"
    );


exampleButtons.forEach(
    function (button) {

        button.addEventListener(
            "click",
            function () {

                reviewInput.value =
                    button.dataset.review;

                reviewInput.focus();

            }
        );

    }
);


// ==========================================
// BUTTON EVENTS
// ==========================================

analyzeBtn.addEventListener(
    "click",
    analyzeReview
);


clearBtn.addEventListener(
    "click",
    clearReview
);


// ==========================================
// CTRL + ENTER
// ==========================================

reviewInput.addEventListener(
    "keydown",
    function (event) {

        if (
            event.ctrlKey &&
            event.key === "Enter"
        ) {

            analyzeReview();

        }

    }
);