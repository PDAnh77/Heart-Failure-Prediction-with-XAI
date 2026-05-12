from typing import Literal
from fastapi import APIRouter, Depends, Query, UploadFile
from services.auth_service import require_roles
from services import dataset_service

router = APIRouter()


@router.post("/upload")
def upload_dataset(file: UploadFile, user=Depends(require_roles(["admin", "user"]))):
    return dataset_service.upload_raw_dataset(file, user["user_id"])


@router.get("/{dataset_id}/summary")
def get_dataset_summary(
    dataset_id: str,
    target_column: str = Query(None),
    owner_id: str = Query(None),
    user=Depends(require_roles(["admin", "user"])),
):
    return dataset_service.get_summary(dataset_id, owner_id, user, target_column)


@router.get("/{dataset_id}/rows")
def get_dataset_rows(
    dataset_id: str,
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    owner_id: str = Query(None),
    user=Depends(require_roles(["admin", "user"])),
):
    return dataset_service.get_rows(dataset_id, owner_id, user, limit, offset)


@router.post("/{dataset_id}/preprocess")
def preprocess_dataset(
    dataset_id: str,
    target_column: str,
    imputation_method: Literal["default", "mice", "mean", "knn"] = Query(
        "default", 
        description="Chọn 'default' (Median/Mode), 'mice' (Mean/Mode <=5% và MICE >5%), 'mean' (Average Mean), hoặc 'knn' (K-Nearest Neighbors)"
    ),
    owner_id: str = Query(None),
    user=Depends(require_roles(["admin", "user"])),
):
    return dataset_service.preprocess(dataset_id, owner_id, user, target_column, imputation_method)


@router.get("/{dataset_id}/download")
def download_dataset(dataset_id: str, owner_id: str = Query(None), file_type: str = Query(None), user=Depends(require_roles(["admin", "user"]))):
    return dataset_service.download(dataset_id, owner_id, user, file_type)


@router.get("/{dataset_id}/eda", dependencies=[Depends(require_roles(["admin", "user"]))])
def get_dataset_eda(
    dataset_id: str,
    target_column: str,
    owner_id: str = Query(None),
    user=Depends(require_roles(["admin", "user"])),
):
    return dataset_service.get_eda(dataset_id, target_column, owner_id, user)


@router.get("/{dataset_id}/feature-selection")
def dataset_feature_selection(
    dataset_id: str,
    target_column: str,
    size: int = Query(80, ge=10, le=200),
    n_gen: int = Query(10, ge=1, le=50),
    mutation_rate: float = Query(0.2, ge=0.01, le=0.5),
    n_parents: int = Query(None),
    owner_id: str = Query(None),
    user=Depends(require_roles(["admin", "user"])),
    model_name: str = Query(None),
    test_size: float = Query(0.3, ge=0.1, le=0.5),
    balancing_method: Literal["none", "adasyn"] = Query("none", description="Chọn phương pháp cân bằng dữ liệu: 'none' (không áp dụng) hoặc 'adasyn' (ADASYN oversampling)"),
):
    return dataset_service.genetic_selection(
        dataset_id, target_column, owner_id, user, size, n_gen, mutation_rate, n_parents, model_name, test_size, balancing_method
    )
