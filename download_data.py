import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader
mySoccerNetDownloader=SoccerNetDownloader(LocalDirectory="soccernet_data")
mySoccerNetDownloader.password = "s0cc3rn3t"
mySoccerNetDownloader.downloadDataTask(task="tracking", split=["train", "test"])
