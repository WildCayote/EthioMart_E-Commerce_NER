import csv, os
from telethon import TelegramClient
from dotenv import load_dotenv


async def scrape_channel(client, channel_username, writer, media_dir):
    """
    This is a function that will write messages found from a telegram channel into a csv file.

    Args:
        client(telethon.TelegramClient): an instance of a telethon TelegramClient instance
        channel_username(string): the username of a telegram channel, starts with @
        writer(csv.writer): an instance of a csv writer
    Returns:
        None
    """
    entity = await client.get_entity(channel_username)
    channel_title = entity.title  # Extract the channel's title
    async for message in client.iter_messages(entity, limit=10000):
        media_path = None
        if message.media and hasattr(message.media, 'photo'):
            # Create a unique filename for the photo
            filename = f"{channel_username}_{message.id}.jpg"
            media_path = os.path.join(media_dir, filename)
            # Download the media to the specified directory if it's a photo
            await client.download_media(message.media, media_path)
        
        # Write the channel title along with other data
        writer.writerow([channel_title, channel_username, message.id, message.message, message.date, media_path])


if __name__ == "__main__":
    ...
