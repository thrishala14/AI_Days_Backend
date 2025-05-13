# from uvicorn import run
# from dotenv import load_dotenv
# import os

# load_dotenv()

# if __name__ == "__main__":
#     run(
#         "app.main:app",
#         reload = True
#     )


import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)