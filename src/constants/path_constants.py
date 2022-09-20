from src.utils.helper import get_project_root


ROOT_DIR = str(get_project_root())
XES_LOG_FOLDER = ROOT_DIR + "/data/input/log_xes"
IN_LOG_PATH = ROOT_DIR + "/data/input/log_xes/5x5_2S.xes"
IN_DECL_PATH = ROOT_DIR + "/data/declare_models/5x5_2S.decl"
#IN_DECL_PATH = ROOT_DIR + "/data/formulas/5x5_1S.decl"
OUT_PDF_DECISION_TREE_PATH = ROOT_DIR + "/data/output/pdf/decision_tree.pdf"