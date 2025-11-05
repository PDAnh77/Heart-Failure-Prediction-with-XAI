from fastapi import APIRouter, HTTPException, Query
from db.database import supabase
from schemas.patient_schema import PatientSchema

router = APIRouter()

TABLE_NAME = "patient_info"

@router.get("/")
def get_patients(limit: int = Query(10, ge=1, le=100, description="Số bản ghi mỗi trang"),
                 offset: int = Query(0, ge=0, description="Vị trí bắt đầu (bỏ qua bao nhiêu bản ghi)")):
    data = supabase.table(TABLE_NAME).select("*").range(offset, offset + limit - 1).execute()
    return {"data": data.data, "count": len(data.data)}

@router.get("/{patient_id}")
def get_patient(patient_id: str):
    data = supabase.table(TABLE_NAME).select("*").eq("id", patient_id).execute()
    if not data.data:
        raise HTTPException(status_code=404, detail="Không tìm thấy bệnh nhân")
    return data.data[0]

@router.post("/")
def create_patient(patient: PatientSchema):
    result = supabase.table(TABLE_NAME).insert(patient.model_dump()).execute()
    return result.data[0]

@router.delete("/{patient_id}")
def delete_patient(patient_id: str):
    result = supabase.table(TABLE_NAME).delete().eq("id", patient_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Không tìm thấy bệnh nhân")
    return {"message" : "Đã xóa bệnh nhân"}
    
    