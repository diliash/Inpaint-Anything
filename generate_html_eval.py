import csv
import math
import os
from glob import glob

FINAL_DIR = "/project/3dlg-hcvc/opmotion/www"

def generate_html(gt_path, results_path, methods, models_per_page=50):
    cur_dir = os.getcwd()
    # Get all model IDs
    model_ids = sorted([d for d in os.listdir(gt_path) if os.path.isdir(os.path.join(gt_path, d))])

    # Calculate number of pages
    total_pages = math.ceil(len(model_ids) / models_per_page)

    # subexp_dir,image_name,psnr,ssim,lpips
    image_metrics_path = f"{results_path}/results.csv"
    # subexp_dir,image_name,rmse,absrel
    dav1_metrics_path = f"{results_path}/results_depth.csv"
    # subexp_dir,image_name,rmse,absrel
    dav2_metrics_path = f"{results_path}/results_depth-v2.csv"

    # Read metrics
    image_metrics = {}
    with open(image_metrics_path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            subexp_dir, image_name, psnr, ssim, lpips = row
            image_metrics[(subexp_dir, image_name)] = (psnr, ssim, lpips)

    dav1_metrics = {}
    with open(dav1_metrics_path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            subexp_dir, image_name, rmse, absrel = row
            dav1_metrics[(subexp_dir, image_name)] = (rmse, absrel)

    dav2_metrics = {}
    with open(dav2_metrics_path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            subexp_dir, image_name, rmse, absrel = row
            dav2_metrics[(subexp_dir, image_name)] = (rmse, absrel)

    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Defurnishing + Depth Evaluation</title>
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
        html_content += "<tr><th>Model ID</th><th>GT</th><th>GT Unfurnished</th>"
        for method in methods:
            html_content += f"<th>{method}</th>"
        html_content += "<th>GT Depth</th>"
        for method in methods:
            html_content += f"<th colspan='2'>{method} Depth</th>"
        html_content += "</tr>\n"
        # Separate subcolumns for Depth Anything v1 and v2
        html_content += "<tr><th></th><th></th><th></th><th></th>"
        for method in methods:
            html_content += "<th></th>"
        for method in methods:
            html_content += "<th>DAv1</th><th>DAv2</th>"
        html_content += "</tr>\n"

        for model_id in page_model_ids:
            html_content += f"<tr><td>{model_id}</td>"

            # GT image
            gt_image_path = f"{gt_path}/{model_id}/scene.png"
            html_content += f'<td><img src="{os.path.relpath(gt_image_path, FINAL_DIR)}" alt="GT {model_id}"></td>'

            # GT Unfurnished image
            gt_unfurnished_image_path = f"{gt_path}/{model_id}/room.png"
            html_content += f'<td><img src="{os.path.relpath(gt_unfurnished_image_path, FINAL_DIR)}" alt="GT Unfurnished {model_id}"></td>'

            # Method images
            for method in methods:
                # Should be local
                method_image_path = f"{results_path}/{method}/{model_id}"
                method_image = next((f for f in os.listdir(method_image_path) if f.endswith('merged.png')), None)
                if method_image:
                    full_path = f"{method_image_path}/{method_image}"
                    html_content += f'<td><img src="{os.path.relpath(full_path, cur_dir)}" alt="{method} {model_id}"></td>'
                else:
                    html_content += '<td>No image found</td>'

            # GT Depth
            gt_depth_path = f"{gt_path}/{model_id}/depth_vis.room.png"
            html_content += f'<td><img src="{os.path.relpath(gt_depth_path, FINAL_DIR)}" alt="GT Depth {model_id}"></td>'

            # Method Depths
            for method in methods:
                # DAv1
                method_depth_path = f"{results_path}/{method}/{model_id}"
                method_depth = next((f for f in os.listdir(method_depth_path) if f.endswith('depth-color.png')), None)
                if method_depth:
                    full_path = f"{method_depth_path}/{method_depth}"
                    html_content += f'<td><img src="{os.path.relpath(full_path, cur_dir)}" alt="{method} Depth {model_id}"></td>'
                else:
                    html_content += '<td>No image found</td>'

                # DAv2
                method_depth_path = f"{results_path}/{method}/{model_id}"
                method_depth = next((f for f in os.listdir(method_depth_path) if f.endswith('depth-color-v2.png')), None)
                if method_depth:
                    full_path = f"{method_depth_path}/{method_depth}"
                    html_content += f'<td><img src="{os.path.relpath(full_path, cur_dir)}" alt="{method} Depth {model_id} v2"></td>'
                else:
                    html_content += '<td>No image found</td>'
            html_content += "</tr>\n"

            # Next row for Metrics
            html_content += "<tr><td></td><td></td><td></td>"  # skip first three columns
            for method in methods:
                subexp_dir = f"{results_path}/{method}"
                image_name = model_id
                psnr, ssim, lpips = image_metrics.get((subexp_dir, image_name), ("N/A", "N/A", "N/A"))
                # Round to 3 decimal places
                psnr, ssim, lpips = map(lambda x: round(float(x), 3), (psnr, ssim, lpips))
                html_content += f"<td>PSNR: {psnr}<br>SSIM: {ssim}<br>LPIPS: {lpips}</td>"
            html_content += "<td></td>"
            for method in methods:
                subexp_dir = f"{results_path}/{method}"
                image_name = model_id
                rmse, absrel = dav1_metrics.get((subexp_dir, image_name), ("N/A", "N/A"))
                rmse_v2, absrel_v2 = dav2_metrics.get((subexp_dir, image_name), ("N/A", "N/A"))
                # Round to 3 decimal places
                rmse, absrel = map(lambda x: round(float(x), 3), (rmse, absrel))
                rmse_v2, absrel_v2 = map(lambda x: round(float(x), 3), (rmse_v2, absrel_v2))
                html_content += f"<td>RMSE: {rmse}<br>AbsRel: {absrel}</td><td>RMSE: {rmse_v2}<br>AbsRel: {absrel_v2}</td>"
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

    with open('inpatint_depth_eval.html', 'w') as f:
        f.write(html_content)

    print(f"HTML file 'inpatint_depth_eval.html' has been generated with {total_pages} pages.")

# Usage

gt_path = "/project/3dlg-hcvc/diorama/wss/scenes"
results_path = "/local-scratch2/diliash/diorama/third_party/Inpaint-Anything/results/gt-rerun"
methods = [path.split("/")[-1] for path in glob(f"{results_path}/*") if os.path.isdir(path)]
generate_html(gt_path, results_path, methods)
