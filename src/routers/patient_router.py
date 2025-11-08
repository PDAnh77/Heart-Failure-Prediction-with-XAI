from fastapi import APIRouter, HTTPException, Query
from db.database import supabase
from schemas.patient_schema import PatientGet, PatientCreate, PatientUpdate

router = APIRouter()
TABLE_NAME = "patient_info"

@router.get("/")
def get_patients(limit: int = Query(10, ge=1, le=100, description="Number of patient records per page"),
                 offset: int = Query(0, ge=0, description="Starting index")):
    result = supabase.table(TABLE_NAME).select("*").range(offset, offset + limit - 1).execute()
    return {"data": result.data, "count": len(result.data)}

@router.get("/{patient_id}", response_model=PatientGet)
def get_patient(patient_id: str):
    result = supabase.table(TABLE_NAME).select("*").eq("id", patient_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Patient not found")
    return result.data

@router.post("/")
def create_patient(new_patient: PatientCreate):
    result = supabase.table(TABLE_NAME).insert(new_patient.model_dump()).execute()
    return result.data

@router.put("/{patient_id}")
def update_patient(patient_id: str, patient: PatientUpdate):
    update_data = patient.model_dump(exclude_unset=True)
    result = supabase.table(TABLE_NAME).update(update_data).eq("id", patient_id).execute()
    return result.data

@router.delete("/{patient_id}")
def delete_patient(patient_id: str):
    result = supabase.table(TABLE_NAME).delete().eq("id", patient_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Patient not found")
    return "Success"
    