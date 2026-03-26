# Credits

This project was developed as part of a larger system involving retrieval-augmented generation (RAG), Azure AI Search, and tool-integrated LLM workflows.

## External Resources and References

### Model Context Protocol (MCP)

Portions of the MCP client and server implementation were adapted from official documentation:

- https://modelcontextprotocol.io/docs/develop/build-server  
- https://modelcontextprotocol.io/docs/develop/build-client  

These resources were used to guide the structure and integration of the MCP server and client components.

---

### Azure AI Search

The design and implementation of Azure AI Search index schemas and related configuration were informed by official Microsoft Azure documentation and examples:

- https://learn.microsoft.com/en-us/azure/search/

These references were used for:
- Index schema design  
- Vector search configuration  
- Field definitions and filtering capabilities  

---

## Libraries and Frameworks

This project makes use of the following open-source libraries and frameworks:

- FastAPI – backend API framework  
- React – frontend user interface  
- Azure SDK for Python – Azure AI Search integration  
- LangExtract – structured metadata extraction  
- LangChain Text Splitters – document chunking utilities  
- Redis – caching and state management  
- Docker – containerisation and deployment  

---

## Acknowledgements

The above resources significantly contributed to the development of this system by providing foundational patterns, implementation guidance, and best practices.