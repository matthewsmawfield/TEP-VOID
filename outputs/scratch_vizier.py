from astroquery.vizier import Vizier
v = Vizier(columns=['*'], row_limit=5)
catalogs = v.get_catalogs('J/A+A/646/A113')
if catalogs:
    print(catalogs[0])
    print(catalogs[0].columns)
