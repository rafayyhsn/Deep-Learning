# ============================================
# TEXT PROMPT TEMPLATES  (used only by baseline MVFA)
# ============================================

TEMPLATES = [
    'a cropped photo of the {}.', 
    'a cropped photo of a {}.', 
    'a close-up photo of a {}.', 
    'a close-up photo of the {}.', 
    'a bright photo of a {}.', 
    'a bright photo of the {}.', 
    'a dark photo of the {}.', 
    'a dark photo of a {}.',
    'a jpeg corrupted photo of a {}.', 
    'a jpeg corrupted photo of the {}.', 
    'a blurry photo of the {}.', 
    'a blurry photo of a {}.',
    'a photo of the {}', 
    'a photo of a {}', 
    'a photo of a small {}', 
    'a photo of the small {}', 
    'a photo of a large {}',
    'a photo of the large {}', 
    'a photo of the {} for visual inspection.', 
    'a photo of a {} for visual inspection.',
    'a photo of the {} for anomaly detection.', 
    'a photo of a {} for anomaly detection.'
]


# ============================================
# REAL NAMES FOR TEMPLATE MODE
# (kept for compatibility — used by original code)
# ============================================

REAL_NAME = {
    'Brain': 'brain MRI',
    'Liver': 'liver CT',
    'Retina_RESC': 'retinal fundus image',
    'Retina_OCT2017': 'retinal OCT scan',
    'Chest': 'chest x-ray',
    'Histopathology': 'histopathology slide'
}


# ============================================
# FULL DOMAIN-AWARE PROMPTS (IMPROVEMENT 1B)
# For direct CLIP encoding — NO templates used
# 
# Each organ has:
#   - normal:   list of healthy descriptions
#   - abnormal: list of diseased descriptions
# ============================================

MEDICAL_PROMPTS = {

    "Liver": {
        "normal": [
            "a CT scan of a healthy liver",
            "a CT abdominal image showing a normal liver",
            "a CT image of a liver without any lesion",
            "a liver CT scan showing no abnormalities",
            "a normal liver in a CT axial slice"
        ],
        "abnormal": [
            "a CT scan of a liver with a hypodense lesion",
            "a CT abdominal image showing a liver tumor",
            "a CT image of a liver with a focal mass",
            "a liver CT scan showing a hyperdense tumor",
            "a CT axial slice of a liver with abnormal lesion"
        ]
    },

    "Brain": {
        "normal": [
            "a T1-weighted MRI scan of the brain without lesions",
            "a brain MRI showing normal anatomy",
            "a brain MRI without pathology",
            "a healthy brain MRI axial slice"
        ],
        "abnormal": [
            "a T1-weighted MRI scan showing a brain tumor",
            "a brain MRI with a hyperintense lesion",
            "a brain MRI showing abnormal mass effect",
            "a brain MRI scan with pathological findings"
        ]
    },

    "Retina_OCT2017": {
        "normal": [
            "an OCT retinal scan with normal macular structure",
            "a retinal OCT without pathology",
            "a healthy OCT retinal scan",
            "a normal OCT image of retinal layers"
        ],
        "abnormal": [
            "an OCT scan showing subretinal fluid",
            "a retinal OCT scan showing retinal lesions",
            "an OCT image with macular degeneration",
            "a retinal OCT with abnormal fluid accumulation"
        ]
    },

    "Retina_RESC": {
        "normal": [
            "a retinal fundus image without lesions",
            "a healthy retinal fundus image",
            "a fundus photograph showing normal vessels",
            "a clean retinal fundus image"
        ],
        "abnormal": [
            "a retinal fundus image showing abnormal exudates",
            "a retinal image with pathological changes",
            "a fundus image showing microaneurysms",
            "a retinal fundus scan with disease signs"
        ]
    },

    "Chest": {
        "normal": [
            "a chest x-ray with clear lungs",
            "a chest radiograph without abnormalities",
            "a normal chest x-ray image",
            "a healthy chest radiograph"
        ],
        "abnormal": [
            "a chest x-ray showing opacities",
            "a chest radiograph showing lesions",
            "a chest x-ray with pathological lung findings",
            "a chest radiograph showing infiltrates"
        ]
    },

    "Histopathology": {
        "normal": [
            "a histopathology slide showing normal tissue",
            "a microscopy image of healthy tissue",
            "a pathology slide with normal cellular structure",
            "a normal tissue biopsy microscopy image"
        ],
        "abnormal": [
            "a histopathology slide showing malignant cells",
            "a microscopy image with abnormal cellular structure",
            "a pathology slide showing neoplastic changes",
            "a tissue biopsy image showing cancerous cells"
        ]
    }
}
