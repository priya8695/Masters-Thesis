# Introduction
Endovascular repair of the thoracic aorta, also referred to as thoracic endovascular aortic repair (TEVAR), refers to a minimally invasive approach that involves placing a stent-graft in the thoracic or thoracoabdominal aorta for the treatment of a variety of thoracic aortic pathologies. In contradiction to open surgery, TEVAR results in reduced recovery times and potentially improved survival rates. Feasibility of TEVAR and correct endograft sizing are based on measurements of Ishimaru’s proximal landing zones. 
However, TEVAR of the aortic arch still carries a significant risk of medium and long-term complications, including endoleak, endograft migration, and collapse.
This may be due to its complex structure and computation of geometric parameters, such as angulation and tortuosity can help to avoid hostile landing zones. 
The primary goal of this project is to segment the aorta from provided CT scan images, map the landzones (Ishimaru’s proximal landing zones Z0, Z1, Z2, and Z3), and compute various geometric parameters of the aortic arch. The pipeline consists of following key processing steps:
# Aorta segmentation from CT scan images:
Utilize a U-Net based deep learning model for aorta segmentation, fine-tuning hyperparameters,
and comparing its performance with other models.
# Triangulated surface mesh generation and centerline extraction:
Generate triangulated surface meshes for the segmented aortic regions and compute centerline
representing the course of the aorta in 3D.
# Landmark detection and landing zone mapping:
Develop an algorithm for identifying Ishimaru’s proximal landing zones Z0, Z1, Z2, and Z3.
Proper selection of the landing zone is crucial for the TEVAR procedure.
# Parameter computation of each zone and aortic arch:
Calculate the arc length, angulation, maximum diameter, and tortuosity angle for each
landing zone, and determine the outer radius of curvature, centerline radius of curvature, and
centerline tortuosity index for the aortic arch.
