import math
import os
from glob import glob


def generate_html(gt_path, results_path, methods, models_per_page=50):
    # Get all model IDs
    model_ids = sorted([d for d in os.listdir(gt_path) if os.path.isdir(os.path.join(gt_path, d))])

    # Calculate number of pages
    total_pages = math.ceil(len(model_ids) / models_per_page)

    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Image Comparison</title>
        <style>
            table {
                border-collapse: collapse;
                width: 100%;
            }
            th, td {
                border: 1px solid black;
                padding: 8px;
                text-align: center;
            }
            img {
                width: 200px;
                height: auto;
            }
            #load-more {
                display: block;
                margin: 20px auto;
                padding: 10px 20px;
                font-size: 16px;
            }
            .page {
                display: none;
            }
            .page:first-child {
                display: table;
            }
        </style>
    </head>
    <body>
    """

    for page in range(total_pages):
        start_index = page * models_per_page
        end_index = min((page + 1) * models_per_page, len(model_ids))
        page_model_ids = model_ids[start_index:end_index]

        html_content += f'<table class="page" id="page-{page + 1}">\n'
        html_content += "<tr><th>Model ID</th><th>GT</th>"
        for method in methods:
            html_content += f"<th>{method}</th>"
        html_content += "</tr>\n"

        for model_id in page_model_ids:
            html_content += f"<tr><td>{model_id}</td>"

            # GT image
            gt_image_path = f"{gt_path}/{model_id}/scene.png"
            html_content += f'<td><img src="{gt_image_path}" alt="GT {model_id}"></td>'

            # Method images
            for method in methods:
                method_image_path = f"./results/gt/{method}/{model_id}"
                method_image = next((f for f in os.listdir(method_image_path) if f.endswith('.png')), None)
                if method_image:
                    full_path = f"{method_image_path}/{method_image}"
                    html_content += f'<td><img src="{full_path}" alt="{method} {model_id}"></td>'
                else:
                    html_content += '<td>No image found</td>'

            html_content += "</tr>\n"

        html_content += "</table>\n"

    html_content += """
    <button id="load-more">Load More</button>

    <script>
    let currentPage = 1;
    const totalPages = """ + str(total_pages) + """;

    document.getElementById('load-more').addEventListener('click', function() {
        if (currentPage < totalPages) {
            currentPage++;
            document.getElementById(`page-${currentPage}`).style.display = 'table';
            if (currentPage === totalPages) {
                this.style.display = 'none';
            }
        }
    });
    </script>

    </body>
    </html>
    """

    with open('image_comparison.html', 'w') as f:
        f.write(html_content)

    print(f"HTML file 'image_comparison.html' has been generated with {total_pages} pages.")

# Usage
gt_path = "/project/3dlg-hcvc/diorama/wss/scenes"
results_path = "./results/gt"
methods = [path.split("/")[-1] for path in glob(f"{results_path}/*")]
generate_html(gt_path, results_path, methods)
