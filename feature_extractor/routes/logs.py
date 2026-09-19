import datetime
from fastapi import APIRouter, status, HTTPException
from fastapi.responses import Response

from feature_extractor.utils import get_logs_zip_file

router = APIRouter(tags=["logs"])


@router.get("/logs", response_class=Response, status_code=status.HTTP_200_OK)
async def download_logs():
    try:
        zip_bytes = await get_logs_zip_file()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    if not zip_bytes:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No log files found")

    zip_name = f"{datetime.datetime.now(datetime.UTC).strftime('%Y%m%d-%H%M%SZ')}.zip"
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{zip_name}"'}
    )
