"""Una ejecución administrada, sin Docker Desktop, sobre rcssserver oficial."""
from .entrenar import parser, train
from .comun.servidor import local_server

if __name__ == '__main__':
    args = parser().parse_args()
    with local_server(args.seed,args.port) as (args.host,args.port):
        train(args)
