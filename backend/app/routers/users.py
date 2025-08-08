import os
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
import psycopg2
from typing import List
from auth_utils import get_current_user
from db import get_database_connection
@router.post('/upload_file')
async def upload_file(
    file: UploadFile = File(...),
    current_user: str = Depends(get_current_user)  # 添加token验证
):
    try:
        file_location = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_location, 'wb') as f:
            content = await file.read()
            f.write(content)
        # 存储文件路径到数据库
        rel_path = f'/{UPLOAD_DIR}/{file.filename}'
        conn = get_database_connection()
        cursor = conn.cursor()
        cursor.execute('INSERT INTO files (file_path) VALUES (%s)', (rel_path,))
        conn.commit()
        conn.close()
        return JSONResponse({
            'success': True,
            'message': '文件上传成功',
            'file_path': rel_path
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'文件上传失败: {str(e)}')

@router.get('/files')
def get_all_files():
    try:
        conn = get_database_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT file_path FROM files ORDER BY uploaded_at DESC')
        files = cursor.fetchall()
        conn.close()
        file_list = [f[0] for f in files]
        return JSONResponse({
            'success': True,
            'files': file_list
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'获取文件列表失败: {str(e)}')
