class FileReaderInput(BaseModel):

    path: str = Field(description="file path")

def read_file(path: str) -> str:
    clean_path = path.strip("\n")
    path = Path(__file__).parent.parent
    with open(os.path.join(f"{path}/polisvergelijker_files", clean_path), "r") as f:
    # with open(os.path.join(f"../polisvergelijker_files", clean_path), "r") as f:
        return f.read()
    
file_reader = StructuredTool.from_function(
    func=read_file,
    name="FileReader",
    description="read files",
    args_schema=FileReaderInput,
    return_direct=True,
)

file_reader_tool = Tool(
        name="Read_file",  # Name of the tool
        func=file_reader,  # Function that the tool will execute
        # Description of the tool
        description="Useful for when you need to know the content of a file.",
    )